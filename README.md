# COVID-19-30-day-Mortality-Prediction-from-CXR

Dataset: https://stanfordmlgroup.github.io/competitions/chexpert/

## Preprocess Data
### Training data and Validation data
由原本給的圖片資料夾，經過分類後用copyfile()把圖片複製到下面四個資料夾中。
依照csv檔裡給的死亡或存活資訊將圖片分類，並且以8:2分為訓練與驗證資料集。
```python=
train_dead_path = './TRAIN/dead/'
train_alive_path = './TRAIN/alive/'
valid_dead_path = './VALID/dead/'
valid_alive_path = './VALID/alive/'
```

```python=
shutil.copyfile(src,dst)
```

### Data augmentation
因為圖片總數量只有約1400張，數量很少，所以我認為使用data augmentation的話可以增加訓練出來的效果，這裡我使用keras裡的ImageDataGenerator()來做圖片增強，改動的參數包含旋轉、平移、縮放、水平翻轉、剪切。

接著，將前面存在Train資料夾裡的training data用flow_from_directory讀入並生成generator，這樣在訓練時，每個epoch都會產生出經過增強後不同的圖片，以彌補原本數量很少的data。

Validation的資料集因為不需要做資料增強所以在ImageDataGenerator()不用增加參數，因此不附上程式碼，Valid資料夾內的data則用相同方式讀進validation_generator。
```python=
from tensorflow.keras.preprocessing import image
train_datagen = image.ImageDataGenerator( 
    rescale = 1./255,
    rotation_range=5,    
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
```
```python=
train_generator = train_datagen.flow_from_directory(
    './TRAIN',
    target_size=IMG_SIZE,
    batch_size=8
)
```
## Build a model-ResNet50
Resnet有個優點就是即使網路加深，準確率卻不會因此下降，且Resnet方便好用，所以我使用pretrained的ResNet50(weights="imagenet")當作我的model。
由於他原本預設的output分類數是1000，所以我在最後面再多加layer讓model變為二分類。
```python=
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
model = ResNet50(
    input_shape = (IMG_SIZE[0],IMG_SIZE[1],3),
    include_top=False,
    weights="imagenet",
    input_tensor=None,
)

NUM_CLASSES = 2
x = model.output
x = Flatten()(x)
x = Dropout(0.2)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
```
在嘗試了三個不同的optimizer後，Adam的效果比SGD與Adadelta好，所以就使用Adam來優化。
根據剛剛得到的generator來訓練model。
```python=
sgd = tf.keras.optimizers.SGD(lr=1e-5)
adam = tf.keras.optimizers.Adam(lr=1e-5)
adadelta = tf.keras.optimizers.Adadelta()
net_final = Model(inputs=model.input, outputs=output_layer)
net_final.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
net_final.fit_generator(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)

```