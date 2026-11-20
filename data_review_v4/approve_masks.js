async function(data) {
 const panel=document.createElement('div');panel.style='padding:12px;background:white;color:#17202a;font:15px sans-serif;max-width:1240px';
 const heading=document.createElement('h3');heading.textContent=data.id+' — '+data.split;panel.appendChild(heading);
 const help=document.createElement('p');help.textContent='Review the green automatic proposal against the photograph. Approve means all visible animals are correctly outlined and background is excluded. Reject keeps the proposal out of verified labels.';panel.appendChild(help);
 const name=document.createElement('input');name.placeholder='Reviewer name (saved for next images)';name.value=data.reviewer;panel.appendChild(name);
 const buttons=document.createElement('div');panel.appendChild(buttons);
 const status=document.createElement('p');status.textContent='Proposed masks: '+data.mask_count+' / human-box animals: '+data.expected;panel.appendChild(status);
 const warnings=document.createElement('p');warnings.textContent=data.warnings.join(' | ');warnings.style='color:#9a5a00';panel.appendChild(warnings);
 const image=document.createElement('img');image.src=data.image;image.style='max-width:100%;display:block';panel.appendChild(image);
 const resize=()=>{if(window.google&&google.colab&&google.colab.output)google.colab.output.setIframeHeight(document.documentElement.scrollHeight,true);};image.onload=resize;
 document.body.appendChild(panel);requestAnimationFrame(resize);
 return new Promise(resolve=>{for(const [label,decision] of [['Approve','approve'],['Reject','reject'],['Close','close']]){const b=document.createElement('button');b.textContent=label;b.style='margin:8px;padding:14px 25px;font-weight:bold';b.disabled=decision==='approve'&&!data.enabled;b.onclick=()=>{if(decision==='close'){panel.remove();resolve(null);return;}if(!name.value.trim()){status.textContent='Enter your reviewer name once.';return;}panel.remove();resolve({decision,reviewer:name.value.trim()});};buttons.appendChild(b);}resize();});
}
