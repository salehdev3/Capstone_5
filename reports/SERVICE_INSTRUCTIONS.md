# Running the service

Open [demo.ipynb in Colab](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/project/demo.ipynb). Mount the team Google Drive containing MyDrive/CameraTrap_Phase2, including the manifest, images and model releases. Run the initialization cell and wait for the release to load. Select cct_test, team_test, photos or video in the second cell and run it. Video mode samples up to five frames at one-second intervals. The notebook writes annotated images, service_test.json and an MP4 under CameraTrap_Phase2/presentation_demos/.

The local FastAPI server provides GET /health and POST /batch. The latter accepts multipart image files as images and a JSON metadata string with one unique id and an event for each image. Timestamp is optional. The response contains frames and events, with boxes, species, masks where available and timings. Invalid input returns HTTP 400. Access to the team Drive and an internet connection are required.
