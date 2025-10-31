from dataset.script import prepare_dataset
from model.autoencoder import Autoencoder
from config import Config
from utils import configure_lr, callbacks,evaluation

if __name__=='__main__':
    config=Config()
    train_data, val_data, test_data = prepare_dataset("dataset/train_ori.mat", "dataset/val_ori.mat", "dataset/test_ori.mat")
    Autoencoder = Autoencoder(config.inbits, config.cbits, config.neurons)
    initial_epoch,model_checkpoint_callback, csv_logger_callback = callbacks(config)
    Autoencoder = configure_lr(config)
    updated_model_history=Autoencoder.fit(x=train_data, y=train_data, batch_size=config.batch_size, epochs=config.epochs, callbacks=[model_checkpoint_callback,csv_logger_callback], validation_data=(val_data, val_data),
                validation_batch_size=1000,initial_epoch=initial_epoch)
    
    result=evaluation(Autoencoder,test_data,config)
    print(result)
    result.to_csv(config.result_path,index=False)

    

    

    
