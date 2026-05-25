# 配置文件
class Config:
    # 数据路径
    DATA_PATH = 'data/fer2013/fer2013.csv'

    # 模型参数
    NUM_CLASSES = 7
    INPUT_SIZE = 48

    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # 类别名称
    CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # 类别权重（处理不平衡）
    CLASS_WEIGHTS = [1.0, 1.5, 1.2, 0.8, 1.0, 1.0, 0.9]