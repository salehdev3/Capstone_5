"""Export reference/prediction overlays and review suggestions; never invent reviewer sign-off."""
from pathlib import Path
import json,hashlib,zipfile,shutil
from datetime import datetime,timezone
import pandas as pd
from PIL import Image,ImageDraw


def export_review(root):
    root=Path(root);release=root/'releases/Segmentation_M4_Guided_Recovery_v2'
    policy_sha=hashlib.sha256((release/'refinement_policy.json').read_bytes()).hexdigest()
    candidates=[p.parent for p in release.glob('evaluations/*/metrics.json') if json.loads(p.read_text()).get('policy_sha256')==policy_sha]
    if not candidates:raise FileNotFoundError('Current v2 evaluation missing')
    evaluation=sorted(candidates)[-1]
    predictions=json.loads((evaluation/'predictions.json').read_text())
    assert predictions['policy_sha256']==policy_sha
    lookup={Path(r['image']).stem:r for r in predictions['results']}
    review=root/'phase2_results/member4_recovery_v2/segmentation_failure_analysis_v2.csv'
    frame=pd.read_csv(review).fillna('');assert frame.id.is_unique and frame.policy_sha256.eq(policy_sha).all()
    protocol=json.loads((evaluation/'protocol.json').read_text())
    folder=root/'phase2_results/member4_recovery_v2/visual_review_exports'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    suggestions=[]
    for row in frame.to_dict('records'):
        key=row['id'];pred=lookup[key];source=root/'images'/(key+'.jpg')
        label=root/'segmentation/labels/test'/(key+'.txt')
        assert source.is_file() and label.is_file(),key
        expected=protocol.get('reference_sha256',{}).get(key)
        if expected:assert hashlib.sha256(label.read_bytes()).hexdigest()==expected,'Reference changed: '+key
        image=Image.open(source).convert('RGBA');w,h=image.size
        reference=[]
        for line in label.read_text().splitlines():
            if line.strip():
                vals=[float(v) for v in line.split()];assert vals[0]==0 and len(vals)>=7 and len(vals)%2==1
                reference.append(list(zip(vals[1::2],vals[2::2])))
        for polygons,color in [(reference,(0,255,0)),([o['polygon_normalized'] for o in pred['objects']],(255,0,0))]:
            layer=Image.new('RGBA',image.size);draw=ImageDraw.Draw(layer)
            for polygon in polygons:
                points=[(float(x*w),float(y*h)) for x,y in polygon]
                if len(points)>=3:
                    draw.polygon(points,fill=color+(45,));draw.line(points+[points[0]],fill=color+(255,),width=3)
            image=Image.alpha_composite(image,layer)
        image=image.convert('RGB');image.thumbnail((1100,850))
        banner=Image.new('RGB',(max(800,image.width),image.height+65),'white');banner.paste(image,(0,65));draw=ImageDraw.Draw(banner)
        draw.text((8,6),key,fill='black')
        draw.text((8,23),f"Reference GREEN | Prediction RED | IoU {float(row['iou']):.3f} | masks {pred['mask_count']} | fallback {pred['fallback_used']}",fill='black')
        draw.text((8,40),'Yellow overlap; review source photo before assigning physical causes.',fill='black')
        banner.save(folder/(key+'_review.jpg'),quality=92)
        shutil.copyfile(source,folder/(key+'_original.jpg'))
        if pred['detector_prompt_count']==0:
            cause='No detector proposal at the fixed confidence floor, so no segmentation mask was produced.'
            improvement='Add verified training examples of missed animals and evaluate detector recall improvements on validation.'
        elif float(row['iou'])==0:
            cause='Accepted mask has zero overlap with the reference. Visual review is needed to identify wrong-object selection or mask localization.'
            improvement='Inspect fallback prompt/object association; improve training examples and validate a revised selection strategy.'
        elif pred['foreground_area_fraction']>.8:
            cause='Accepted foreground covers over 80% of the image despite very low reference overlap; an oversized mask dominates the result.'
            improvement='Review the oversized detector prompt and SAM output; test prompt-support or rejection rules on validation.'
        else:
            cause='Predicted and reference foreground disagree in extent or placement; physical causes require visual inspection.'
            improvement='Inspect boundary errors, occlusion and missing instances; improve matched training examples and validate any changes.'
        suggestions.append(dict(**row,detector_prompts=pred['detector_prompt_count'],accepted_masks=pred['mask_count'],fallback_used=pred['fallback_used'],
            predicted_area_fraction=pred['foreground_area_fraction'],suggested_cause=cause,suggested_improvement=improvement))
    pd.DataFrame(suggestions).to_csv(folder/'review_suggestions.csv',index=False)
    shutil.copyfile(review,folder/'segmentation_failure_analysis_v2.csv')
    (folder/'README.txt').write_text('12 selected lower-IoU cases, including two above 0.70. GREEN=reference, RED=prediction. Original images included. Suggestions are operational observations, not established physical causes. Review images, fill cause/improvement and use the actual reviewer name. Original audit CSV and references were not modified. No training or inference rerun.')
    archive=folder.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in folder.iterdir():z.write(path,path.name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('Review images and suggestions saved:',archive,flush=True)
    return archive
