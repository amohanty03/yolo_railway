from ultralytics import YOLO

model = YOLO('best_model.pt')

# this will run inference on every image in images/test/
# and write annotated images into predictions/images
model.predict(
    source       = 'images/test',
    save         = True,                    # save the drawn images
    project      = 'predictions',           # root output folder
    name         = 'images',                # subfolder name
    exist_ok     = True                     # overwrite if exists
)
