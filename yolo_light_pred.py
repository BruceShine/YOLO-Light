from ultralytics import YOLO


data = 'digits'
yolo_light = './results/1cv10/digits-0702-S/genotype-16-1903-S.yaml'
name = 'v10_'+ data

data_train = './data/' + data + '.yaml'
data_pred = './data/' + data +  '/test/images'
project = 'yolo_light_pred'

# Create a lightweight YOLO model from scratch
model = YOLO(yolo_light)

# Train the model using rf100 dataset
model.train(data=data_train, epochs=300, batch=32, workers=8, device=[0, 1],
            project=project, name=name, patience=50,
            imgsz=640, val=False, cache=True, exist_ok=True, plots=False)

# Evaluate the model's performance on the validation set
val_model = YOLO('./' + project + '/' + name + '/weights/best.pt')
val_model.val(project=project, name=name+'_val', batch=32, workers=8,
              device=[0, 1], cache=True, exist_ok=True)

# Perform object detection on an image using the model
val_model.predict(source=data_pred, save=True, project=project, name=name+'_pred')


