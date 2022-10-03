
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import numpy as np
from params import *
import pickle
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

PATH = "./"


with open('./train_input.pkl','rb') as f:
    train_input = pickle.load(f)
    print(train_input.shape)
    
with open('./train_target.pkl','rb') as f:
    train_target = pickle.load(f)
    print(train_target.shape)
    
with open('./val_input.pkl','rb') as f:
    val_input = pickle.load(f)
    print(val_input.shape)
    
with open('./val_target.pkl','rb') as f:
    val_target = pickle.load(f)
    print(val_target.shape)    
    
with open('./test_target.pkl','rb') as f:
    test_target = pickle.load(f)
    print(test_target.shape)    

with open('./test_input.pkl','rb') as f:
    test_input = pickle.load(f)
    print(test_input.shape)        

print(train_target.min(),train_target.max())
print(val_target.min(),val_target.max())


# Normalize the target values of the neural network in the interval [-1,1]
train_target -= -0.1711
train_target /= 0.1711*2
train_target *= 2
train_target -= 1


val_target -= -0.1711
val_target /= 0.1711*2
val_target *= 2
val_target -= 1

print(train_target.min(),train_target.max())
print(val_target.min(),val_target.max())


OUTPUT_CHANNELS = 3

def downsample(entered_input,filters, size, apply_batchnorm=True,strides=2):
  
  conv1 = tf.keras.layers.Conv1D(filters, size, strides=strides, padding='same',use_bias=False)(entered_input) 
  conv1 = tf.keras.layers.LeakyReLU()(conv1)
  
  if apply_batchnorm:
    conv1 = tf.keras.layers.BatchNormalization()(conv1)

  return conv1


def upsample(entered_input,filters, size, skip_layer, apply_dropout=False, strides=2, apply_skip=True):
  tran1 = tf.keras.layers.Conv1DTranspose(filters, size, strides=strides, padding='same', use_bias=True)(entered_input)
  tran1 = tf.keras.layers.ReLU()(tran1) 
  if apply_dropout:
      tran1 = tf.keras.layers.Dropout(0.5)(tran1)
  
  if apply_skip:
      tran1 = tf.keras.layers.Concatenate()([tran1,skip_layer])
  return tran1

# Create the Convolutional Neural Network (CNN)
def Generator(input_size): 
  input1 = tf.keras.layers.Input(input_size)  
  output1 = downsample(input1, 64, 3)
  output2 = downsample(output1, 128, 3)
  output3 = downsample(output2, 256, 3)  
  output4 = downsample(output3, 512, 3) 
  output5 = downsample(output4, 512, 3) 
  
  output = upsample(output5, 512, 3, output4, apply_dropout=True)
  output = upsample(output, 256, 3, output3, apply_dropout=False)
  output = upsample(output, 128, 3, output2, apply_dropout=False)
  output = upsample(output, 64, 3, output1, apply_dropout=False)
  
  output = tf.keras.layers.Conv1DTranspose(64, 3, strides=2, padding="same",  activation="relu")(output)
  out = tf.keras.layers.Conv1DTranspose(3, 3, strides=1, padding="same",  activation="tanh")(output)

  model = tf.keras.models.Model(input1,out)
  model.compile(optimizer="adam", loss=custom18(input1),experimental_run_tf_function=False, metrics = [bond_angle_loss(input1), dihedral_loss(input1) , bond_length_loss, MAE, v_loss])
  return model



MAE = tf.keras.losses.MeanAbsoluteError()
MSE = tf.keras.losses.MeanSquaredError()

def bond_length_loss(y_true,y_pred):
   true_length = tf.sqrt(tf.reduce_sum(tf.square(y_true),axis=-1))
   pred_length = tf.sqrt(tf.reduce_sum(tf.square(y_pred),axis=-1))  
   return MAE(true_length,pred_length)

def bond_lengths(y):
   length = tf.sqrt(tf.reduce_sum(tf.square(y),axis=-1))
   return length



def bond_angle_loss(input1):
  def loss(y_pred,y_true):  
    ind_b3 = tf.where(tf.equal(input1[:,:,-3],1))[2::4]
    ind_b1 = tf.where(tf.equal(input1[:,:,-3],1))[0::4]
    
    y_true_new = tf.tensor_scatter_nd_update(y_true, ind_b3, tf.gather_nd(y_true,ind_b1))
    y_pred_new = tf.tensor_scatter_nd_update(y_pred, ind_b3, tf.gather_nd(y_pred,ind_b1))
    
    length_true_1 = bond_lengths(y_true_new)
    length_true_2 = bond_lengths(tf.roll(y_true,shift=-1,axis=-2))
    length_true = tf.math.multiply(length_true_1, length_true_2)
    length_pred_1 = bond_lengths(y_pred_new)
    length_pred_2 = bond_lengths(tf.roll(y_pred,shift=-1,axis=-2))
    length_pred = tf.math.multiply(length_pred_1, length_pred_2)
    
    new_true = tf.math.multiply(y_true_new, tf.roll(y_true,shift=-1,axis=-2))
    true_dot = -tf.reduce_sum(new_true,axis=-1)
    new_pred = tf.math.multiply(y_pred_new, tf.roll(y_pred,shift=-1,axis=-2))
    pred_dot = -tf.reduce_sum(new_pred,axis=-1)
    
    angle_true = tf.math.divide_no_nan(true_dot,length_true)
    angle_pred = tf.math.divide_no_nan(pred_dot,length_pred)
    return MAE(angle_true,angle_pred)
  return loss

def dihedral_loss(input1):
  def d_loss(y_true,y_pred):  
    ind_b3 = tf.where(tf.equal(input1[:,:,-3],1))[2::4]
    ind_b2 = tf.where(tf.equal(input1[:,:,-3],1))[1::4]
    ind_b1 = tf.where(tf.equal(input1[:,:,-3],1))[0::4]
 
    y_true_1 = tf.tensor_scatter_nd_update(y_true, ind_b2, tf.gather_nd(y_true,ind_b3))   
    y_true_1 = tf.tensor_scatter_nd_update(y_true_1, ind_b3, tf.gather_nd(y_true,ind_b2)) 
    y_true_2 = tf.roll(y_true,shift=-1,axis=-2)
    y_true_2 = tf.tensor_scatter_nd_update(y_true_2, ind_b2, tf.gather_nd(y_true_2,ind_b1))
    y_true_3 = tf.roll(y_true,shift=-2,axis=-2)
    
    y_true_1s = tf.concat([tf.gather_nd(tf.roll(y_true,shift=1,axis=-2),ind_b1),tf.gather_nd(y_true,ind_b1)],axis=-2)
    y_true_2s = tf.concat([tf.gather_nd(y_true,ind_b1),tf.gather_nd(y_true_2,ind_b3)],axis=-2)
    y_true_3s = tf.concat([tf.gather_nd(y_true_2,ind_b3),tf.gather_nd(y_true_3,ind_b3)],axis=-2)
    
    cross_true_1 = tf.linalg.cross(y_true_1,y_true_2)
    cross_true_2 = tf.linalg.cross(y_true_2,y_true_3)
    
    cross_true_1s = tf.linalg.cross(y_true_1s,y_true_2s)
    cross_true_2s = tf.linalg.cross(y_true_2s,y_true_3s)    
    
    length_true_1 = bond_lengths(cross_true_1)
    length_true_2 = bond_lengths(cross_true_2)
    length_true = tf.math.multiply(length_true_1, length_true_2)
    
    length_true_1s = bond_lengths(cross_true_1s)
    length_true_2s = bond_lengths(cross_true_2s)
    length_true_s = tf.math.multiply(length_true_1s, length_true_2s)
    
    new_true = tf.math.multiply(cross_true_1, cross_true_2)
    true_dot = tf.reduce_sum(new_true,axis=-1)
    
    new_true_s = tf.math.multiply(cross_true_1s, cross_true_2s)
    true_dot_s = tf.reduce_sum(new_true_s,axis=-1)    
    
    angle_true = tf.math.divide_no_nan(true_dot,length_true)
    angle_true_s = tf.math.divide_no_nan(true_dot_s,length_true_s)
    
    y_pred_1 = tf.tensor_scatter_nd_update(y_pred, ind_b2, tf.gather_nd(y_pred,ind_b3))   
    y_pred_1 = tf.tensor_scatter_nd_update(y_pred_1, ind_b3, tf.gather_nd(y_pred,ind_b2)) 
    y_pred_2 = tf.roll(y_pred,shift=-1,axis=-2)
    y_pred_2 = tf.tensor_scatter_nd_update(y_pred_2, ind_b2, tf.gather_nd(y_pred_2,ind_b1))
    y_pred_3 = tf.roll(y_pred,shift=-2,axis=-2)
    
    y_pred_1s = tf.concat([tf.gather_nd(tf.roll(y_pred,shift=1,axis=-2),ind_b1),tf.gather_nd(y_pred,ind_b1)],axis=-2)
    y_pred_2s = tf.concat([tf.gather_nd(y_pred,ind_b1),tf.gather_nd(y_pred_2,ind_b3)],axis=-2)
    y_pred_3s = tf.concat([tf.gather_nd(y_pred_2,ind_b3),tf.gather_nd(y_pred_3,ind_b3)],axis=-2)
    
    cross_pred_1 = tf.linalg.cross(y_pred_1,y_pred_2)
    cross_pred_2 = tf.linalg.cross(y_pred_2,y_pred_3)
    
    cross_pred_1s = tf.linalg.cross(y_pred_1s,y_pred_2s)
    cross_pred_2s = tf.linalg.cross(y_pred_2s,y_pred_3s)    
    
    length_pred_1 = bond_lengths(cross_pred_1)
    length_pred_2 = bond_lengths(cross_pred_2)
    length_pred = tf.math.multiply(length_pred_1, length_pred_2)
    
    length_pred_1s = bond_lengths(cross_pred_1s)
    length_pred_2s = bond_lengths(cross_pred_2s)
    length_pred_s = tf.math.multiply(length_pred_1s, length_pred_2s)
    
    new_pred = tf.math.multiply(cross_pred_1, cross_pred_2)
    pred_dot = tf.reduce_sum(new_pred,axis=-1)
    
    new_pred_s = tf.math.multiply(cross_pred_1s, cross_pred_2s)
    pred_dot_s = tf.reduce_sum(new_pred_s,axis=-1)    
    
    angle_pred = tf.math.divide_no_nan(pred_dot,length_pred)
    angle_pred_s = tf.math.divide_no_nan(pred_dot_s,length_pred_s)    
    
    return  MAE(angle_true,angle_pred) + MAE(angle_true_s,angle_pred_s) 
  return d_loss





def custom18(input1):
  def loss(y_true,y_pred): 
    lambda_bl = 5
    lambda_ba = 0.6    
    # Compute dihedral angles loss
    ind_b3 = tf.where(tf.equal(input1[:,:,-3],1))[2::4]
    ind_b2 = tf.where(tf.equal(input1[:,:,-3],1))[1::4]
    ind_b1 = tf.where(tf.equal(input1[:,:,-3],1))[0::4]
 
    y_true_1 = tf.tensor_scatter_nd_update(y_true, ind_b2, tf.gather_nd(y_true,ind_b3))   
    y_true_1 = tf.tensor_scatter_nd_update(y_true_1, ind_b3, tf.gather_nd(y_true,ind_b2)) 
    y_true_2 = tf.roll(y_true,shift=-1,axis=-2)
    y_true_2 = tf.tensor_scatter_nd_update(y_true_2, ind_b2, tf.gather_nd(y_true_2,ind_b1))
    y_true_3 = tf.roll(y_true,shift=-2,axis=-2)
    
    y_true_1s = tf.concat([tf.gather_nd(tf.roll(y_true,shift=1,axis=-2),ind_b1),tf.gather_nd(y_true,ind_b1)],axis=-2)
    y_true_2s = tf.concat([tf.gather_nd(y_true,ind_b1),tf.gather_nd(y_true_2,ind_b3)],axis=-2)
    y_true_3s = tf.concat([tf.gather_nd(y_true_2,ind_b3),tf.gather_nd(y_true_3,ind_b3)],axis=-2)
    
    cross_true_1 = tf.linalg.cross(y_true_1,y_true_2)
    cross_true_2 = tf.linalg.cross(y_true_2,y_true_3)
    
    cross_true_1s = tf.linalg.cross(y_true_1s,y_true_2s)
    cross_true_2s = tf.linalg.cross(y_true_2s,y_true_3s)    
    
    length_true_1 = bond_lengths(cross_true_1)
    length_true_2 = bond_lengths(cross_true_2)
    length_true = tf.math.multiply(length_true_1, length_true_2)
    
    length_true_1s = bond_lengths(cross_true_1s)
    length_true_2s = bond_lengths(cross_true_2s)
    length_true_s = tf.math.multiply(length_true_1s, length_true_2s)
    
    new_true = tf.math.multiply(cross_true_1, cross_true_2)
    true_dot = tf.reduce_sum(new_true,axis=-1)
    
    new_true_s = tf.math.multiply(cross_true_1s, cross_true_2s)
    true_dot_s = tf.reduce_sum(new_true_s,axis=-1)    
    
    angle_true = tf.math.divide_no_nan(true_dot,length_true)
    angle_true_s = tf.math.divide_no_nan(true_dot_s,length_true_s)
    
    y_pred_1 = tf.tensor_scatter_nd_update(y_pred, ind_b2, tf.gather_nd(y_pred,ind_b3))   
    y_pred_1 = tf.tensor_scatter_nd_update(y_pred_1, ind_b3, tf.gather_nd(y_pred,ind_b2)) 
    y_pred_2 = tf.roll(y_pred,shift=-1,axis=-2)
    y_pred_2 = tf.tensor_scatter_nd_update(y_pred_2, ind_b2, tf.gather_nd(y_pred_2,ind_b1))
    y_pred_3 = tf.roll(y_pred,shift=-2,axis=-2)
    
    y_pred_1s = tf.concat([tf.gather_nd(tf.roll(y_pred,shift=1,axis=-2),ind_b1),tf.gather_nd(y_pred,ind_b1)],axis=-2)
    y_pred_2s = tf.concat([tf.gather_nd(y_pred,ind_b1),tf.gather_nd(y_pred_2,ind_b3)],axis=-2)
    y_pred_3s = tf.concat([tf.gather_nd(y_pred_2,ind_b3),tf.gather_nd(y_pred_3,ind_b3)],axis=-2)
    
    cross_pred_1 = tf.linalg.cross(y_pred_1,y_pred_2)
    cross_pred_2 = tf.linalg.cross(y_pred_2,y_pred_3)
    
    cross_pred_1s = tf.linalg.cross(y_pred_1s,y_pred_2s)
    cross_pred_2s = tf.linalg.cross(y_pred_2s,y_pred_3s)    
    
    length_pred_1 = bond_lengths(cross_pred_1)
    length_pred_2 = bond_lengths(cross_pred_2)
    length_pred = tf.math.multiply(length_pred_1, length_pred_2)
    
    length_pred_1s = bond_lengths(cross_pred_1s)
    length_pred_2s = bond_lengths(cross_pred_2s)
    length_pred_s = tf.math.multiply(length_pred_1s, length_pred_2s)
    
    new_pred = tf.math.multiply(cross_pred_1, cross_pred_2)
    pred_dot = tf.reduce_sum(new_pred,axis=-1)
    
    new_pred_s = tf.math.multiply(cross_pred_1s, cross_pred_2s)
    pred_dot_s = tf.reduce_sum(new_pred_s,axis=-1)    
    
    angle_pred = tf.math.divide_no_nan(pred_dot,length_pred)
    angle_pred_s = tf.math.divide_no_nan(pred_dot_s,length_pred_s)    

    
    # Compute bond angles loss
    ya_true_new = tf.tensor_scatter_nd_update(y_true, ind_b3, tf.gather_nd(y_true,ind_b1))
    ya_pred_new = tf.tensor_scatter_nd_update(y_pred, ind_b3, tf.gather_nd(y_pred,ind_b1))
    
    length_true_1a = bond_lengths(ya_true_new)
    length_true_2a = bond_lengths(tf.roll(y_true,shift=-1,axis=-2))
    length_true_ba = tf.math.multiply(length_true_1a, length_true_2a)
    length_pred_1a = bond_lengths(ya_pred_new)
    length_pred_2a = bond_lengths(tf.roll(y_pred,shift=-1,axis=-2))
    length_pred_ba = tf.math.multiply(length_pred_1a, length_pred_2a)
    
    new_true_ba = tf.math.multiply(ya_true_new, tf.roll(y_true,shift=-1,axis=-2))
    true_dot_ba = -tf.reduce_sum(new_true_ba,axis=-1)
    new_pred_ba = tf.math.multiply(ya_pred_new, tf.roll(y_pred,shift=-1,axis=-2))
    pred_dot_ba = -tf.reduce_sum(new_pred_ba,axis=-1)
    
    bond_angle_true = tf.math.divide_no_nan(true_dot_ba,length_true_ba)
    bond_angle_pred = tf.math.divide_no_nan(pred_dot_ba,length_pred_ba)
    
    return  lambda_ba*(MAE(angle_true,angle_pred) + MAE(angle_true_s,angle_pred_s)) + MAE(y_pred,y_true) + lambda_bl*bond_length_loss(y_true, y_pred) +  lambda_ba*MAE(bond_angle_true,bond_angle_pred) + v_loss(y_true,y_pred)
  return  loss


    
def v_loss(y_true,y_pred):
    true_tensor = tf.roll(y_true,shift=-3,axis=-2)
    true_tensor1 = tf.roll(y_true,shift=-4,axis=-2)
    true_tensor2 = tf.roll(y_true,shift=-5,axis=-2)
    true_tensor3 = tf.roll(y_true,shift=-6,axis=-2)

    pred_tensor = tf.roll(y_pred,shift=-3,axis=-2)
    pred_tensor1 = tf.roll(y_pred,shift=-4,axis=-2)
    pred_tensor2 = tf.roll(y_pred,shift=-5,axis=-2)
    pred_tensor3 = tf.roll(y_pred,shift=-6,axis=-2)
    
    mask = np.array([True, False, False, False]*(nmer+2))
    
    true_tensor = tf.boolean_mask(true_tensor, mask, axis=-2)
    true_tensor1 = tf.boolean_mask(true_tensor1, mask, axis=-2)
    true_tensor2 = tf.boolean_mask(true_tensor2, mask, axis=-2)
    true_tensor3 = tf.boolean_mask(true_tensor3, mask, axis=-2)
    
    pred_tensor = tf.boolean_mask(pred_tensor, mask, axis=-2)
    pred_tensor1 = tf.boolean_mask(pred_tensor1, mask, axis=-2)
    pred_tensor2 = tf.boolean_mask(pred_tensor2, mask, axis=-2)
    pred_tensor3 = tf.boolean_mask(pred_tensor3, mask, axis=-2)


    v_true = true_tensor*masses_cis[0] + (true_tensor+true_tensor1)*masses_cis[1] + (true_tensor+true_tensor1+true_tensor2)*masses_cis[2] + (true_tensor+true_tensor1+true_tensor2+true_tensor3)*masses_cis[3]
    v_true /= -np.sum(masses_cis)

    v_pred = pred_tensor*masses_cis[0] + (pred_tensor+pred_tensor1)*masses_cis[1] + (pred_tensor+pred_tensor1+pred_tensor2)*masses_cis[2] + (pred_tensor+pred_tensor1+pred_tensor2+pred_tensor3)*masses_cis[3]
    v_pred /= -np.sum(masses_cis)
     
    return MAE(v_true,v_pred)



input_size = (INPUT_WIDTH,6)

model = Generator(input_size)
model.summary()



# Model weights are saved at the end of every epoch, if it's the best seen so far, based on the validation set.
checkpoint_filepath = './Best_model_weights.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss', verbose=1,
    mode='min',
    save_best_only=True)


checkpoint_path = "./tmp/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Model weights are saved at the end of every epoch
model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
   filepath = checkpoint_path, monitor = 'val_loss',
   verbose=1, save_weights_only=True, mode = "min", 
   save_freq='epoch', save_best_only=False)


learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.8, min_lr=0.000001)



history = model.fit(x=train_input,y=train_target,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,verbose=1,
    shuffle=True,validation_data=(val_input,val_target),
    callbacks=[model_checkpoint_callback,model_checkpoint_callback2,learning_rate_reduction])

# Plot the loss function of the validation and training set as a function of epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("loss_plot.png")

# Save the losses file
with open('losses.pkl','wb') as f:
      pickle.dump(history.history, f)



"""## Decoding """

"""## Testing on the entire test dataset"""
# Run the trained model on the entire test dataset


def COM(cords,masses):
    com=[0.0,0.0,0.0]
    totmass = np.sum(masses)
    for i in range(cords.shape[0]):
      com[0] += masses[i]*cords[i][0]
      com[1] += masses[i]*cords[i][1]
      com[2] += masses[i]*cords[i][2]
    com /= totmass
    return com


# Decoding the output of the neural network for each CG type

def cis_d(pos,masses,iosx,iosy,iosz,b0,nmer_count,Coords,identities,PartIndx):          
     b1 = [pos[0,0],pos[0,1],pos[0,2]]
     b2 = [pos[1,0],pos[1,1],pos[1,2]]
     b3 = [pos[2,0],pos[2,1],pos[2,2]]
     b0[nmer_count][:] = [pos[3,0],pos[3,1],pos[3,2]]
     vcoords = np.zeros((5,3))
     totmass = np.sum(masses)
     if nmer_count == 0:  
       posx=-(masses[1]*b1[0]+masses[2]*(b1[0]+b2[0])+masses[3]*(b1[0]+b2[0]+b3[0]))/totmass
       posy=-(masses[1]*b1[1]+masses[2]*(b1[1]+b2[1])+masses[3]*(b1[1]+b2[1]+b3[1]))/totmass
       posz=-(masses[1]*b1[2]+masses[2]*(b1[2]+b2[2])+masses[3]*(b1[2]+b2[2]+b3[2]))/totmass
      
       vcoords[1][:] = [posx,posy,posz]
       vcoords[2][:] = [vcoords[1][0]+b1[0],vcoords[1][1]+b1[1],vcoords[1][2]+b1[2]]     
       vcoords[3][:] = [vcoords[2][0]+b2[0],vcoords[2][1]+b2[1],vcoords[2][2]+b2[2]]    
       vcoords[4][:] = [vcoords[3][0]+b3[0],vcoords[3][1]+b3[1],vcoords[3][2]+b3[2]]      
     
     elif nmer_count != 0: 
       posx=-(b0[nmer_count-1][0]*masses[0]+masses[1]*(b0[nmer_count-1][0]+b1[0])+masses[2]*(b0[nmer_count-1][0]+b1[0]+b2[0])+masses[3]*(b0[nmer_count-1][0]+b1[0]+b2[0]+b3[0]))/totmass
       posy=-(b0[nmer_count-1][1]*masses[0]+masses[1]*(b0[nmer_count-1][1]+b1[1])+masses[2]*(b0[nmer_count-1][1]+b1[1]+b2[1])+masses[3]*(b0[nmer_count-1][1]+b1[1]+b2[1]+b3[1]))/totmass
       posz=-(b0[nmer_count-1][2]*masses[0]+masses[1]*(b0[nmer_count-1][2]+b1[2])+masses[2]*(b0[nmer_count-1][2]+b1[2]+b2[2])+masses[3]*(b0[nmer_count-1][2]+b1[2]+b2[2]+b3[2]))/totmass  
       
       vcoords[0][:] = [posx,posy,posz]
       vcoords[1][:] = [vcoords[0][0]+b0[nmer_count-1][0],vcoords[0][1]+b0[nmer_count-1][1],vcoords[0][2]+b0[nmer_count-1][2]]   
       vcoords[2][:] = [vcoords[1][0]+b1[0],vcoords[1][1]+b1[1],vcoords[1][2]+b1[2]]  
       vcoords[3][:] = [vcoords[2][0]+b2[0],vcoords[2][1]+b2[1],vcoords[2][2]+b2[2]]  
       vcoords[4][:] = [vcoords[3][0]+b3[0],vcoords[3][1]+b3[1],vcoords[3][2]+b3[2]]  
         
     Coords[PartIndx][:] = [iosx+vcoords[1][0],iosy+vcoords[1][1],iosz+vcoords[1][2]]
     Coords[PartIndx+1][:] = [iosx+vcoords[2][0],iosy+vcoords[2][1],iosz+vcoords[2][2]]
     Coords[PartIndx+2][:] = [iosx+vcoords[3][0],iosy+vcoords[3][1],iosz+vcoords[3][2]]
     Coords[PartIndx+3][:] = [iosx+vcoords[4][0],iosy+vcoords[4][1],iosz+vcoords[4][2]]     

     return 

def trans_d(pos,masses,iosx,iosy,iosz,b0,nmer_count,Coords,identities,PartIndx):  
     b1 = [pos[0,0],pos[0,1],pos[0,2]]
     b2 = [pos[1,0],pos[1,1],pos[1,2]]
     b3 = [pos[2,0],pos[2,1],pos[2,2]]
     b0[nmer_count][:] = [pos[3,0],pos[3,1],pos[3,2]]
     vcoords = np.zeros((5,3))
     totmass = np.sum(masses)
     if nmer_count == 0:  
       posx=-(masses[1]*b1[0]+masses[2]*(b1[0]+b2[0])+masses[3]*(b1[0]+b2[0]+b3[0]))/totmass
       posy=-(masses[1]*b1[1]+masses[2]*(b1[1]+b2[1])+masses[3]*(b1[1]+b2[1]+b3[1]))/totmass
       posz=-(masses[1]*b1[2]+masses[2]*(b1[2]+b2[2])+masses[3]*(b1[2]+b2[2]+b3[2]))/totmass
      
       vcoords[1][:] = [posx,posy,posz]
       vcoords[2][:] = [vcoords[1][0]+b1[0],vcoords[1][1]+b1[1],vcoords[1][2]+b1[2]]     
       vcoords[3][:] = [vcoords[2][0]+b2[0],vcoords[2][1]+b2[1],vcoords[2][2]+b2[2]]    
       vcoords[4][:] = [vcoords[3][0]+b3[0],vcoords[3][1]+b3[1],vcoords[3][2]+b3[2]]
     
     elif nmer_count != 0: 
       posx=-(b0[nmer_count-1][0]*masses[0]+masses[1]*(b0[nmer_count-1][0]+b1[0])+masses[2]*(b0[nmer_count-1][0]+b1[0]+b2[0])+masses[3]*(b0[nmer_count-1][0]+b1[0]+b2[0]+b3[0]))/totmass
       posy=-(b0[nmer_count-1][1]*masses[0]+masses[1]*(b0[nmer_count-1][1]+b1[1])+masses[2]*(b0[nmer_count-1][1]+b1[1]+b2[1])+masses[3]*(b0[nmer_count-1][1]+b1[1]+b2[1]+b3[1]))/totmass
       posz=-(b0[nmer_count-1][2]*masses[0]+masses[1]*(b0[nmer_count-1][2]+b1[2])+masses[2]*(b0[nmer_count-1][2]+b1[2]+b2[2])+masses[3]*(b0[nmer_count-1][2]+b1[2]+b2[2]+b3[2]))/totmass  
       
       vcoords[0][:] = [posx,posy,posz]
       vcoords[1][:] = [vcoords[0][0]+b0[nmer_count-1][0],vcoords[0][1]+b0[nmer_count-1][1],vcoords[0][2]+b0[nmer_count-1][2]]   
       vcoords[2][:] = [vcoords[1][0]+b1[0],vcoords[1][1]+b1[1],vcoords[1][2]+b1[2]]  
       vcoords[3][:] = [vcoords[2][0]+b2[0],vcoords[2][1]+b2[1],vcoords[2][2]+b2[2]]  
       vcoords[4][:] = [vcoords[3][0]+b3[0],vcoords[3][1]+b3[1],vcoords[3][2]+b3[2]]  
         
     Coords[PartIndx][:] = [iosx+vcoords[1][0],iosy+vcoords[1][1],iosz+vcoords[1][2]]
     Coords[PartIndx+1][:] = [iosx+vcoords[2][0],iosy+vcoords[2][1],iosz+vcoords[2][2]]
     Coords[PartIndx+2][:] = [iosx+vcoords[3][0],iosy+vcoords[3][1],iosz+vcoords[3][2]]
     Coords[PartIndx+3][:] = [iosx+vcoords[4][0],iosy+vcoords[4][1],iosz+vcoords[4][2]]     

     return 
     
def vinyl_d(pos,masses,iosx,iosy,iosz,b0,nmer_count,Coords,identities,PartIndx):  
     b1 = [pos[0,0],pos[0,1],pos[0,2]]
     b2 = [pos[1,0],pos[1,1],pos[1,2]]
     b3 = [pos[2,0],pos[2,1],pos[2,2]]
     b0[nmer_count][:] = [pos[3,0],pos[3,1],pos[3,2]]
     vcoords = np.zeros((5,3))
     totmass = np.sum(masses)
     if nmer_count == 0:  
       posx=-(masses[1]*b1[0]+masses[2]*(b1[0]+b2[0])+masses[3]*(b1[0]+b2[0]+b3[0]))/totmass
       posy=-(masses[1]*b1[1]+masses[2]*(b1[1]+b2[1])+masses[3]*(b1[1]+b2[1]+b3[1]))/totmass
       posz=-(masses[1]*b1[2]+masses[2]*(b1[2]+b2[2])+masses[3]*(b1[2]+b2[2]+b3[2]))/totmass
      
       vcoords[1][:] = [posx,posy,posz]
       vcoords[2][:] = [vcoords[1][0]+b1[0],vcoords[1][1]+b1[1],vcoords[1][2]+b1[2]]     
       vcoords[3][:] = [vcoords[2][0]+b2[0],vcoords[2][1]+b2[1],vcoords[2][2]+b2[2]]    
       vcoords[4][:] = [vcoords[3][0]+b3[0],vcoords[3][1]+b3[1],vcoords[3][2]+b3[2]]
       
     elif nmer_count != 0: 
       posx=-(b0[nmer_count-1][0]*masses[0]+masses[1]*(b0[nmer_count-1][0]+b1[0])+masses[2]*(b0[nmer_count-1][0]+b1[0]+b2[0])+masses[3]*(b0[nmer_count-1][0]+b1[0]+b2[0]+b3[0]))/totmass
       posy=-(b0[nmer_count-1][1]*masses[0]+masses[1]*(b0[nmer_count-1][1]+b1[1])+masses[2]*(b0[nmer_count-1][1]+b1[1]+b2[1])+masses[3]*(b0[nmer_count-1][1]+b1[1]+b2[1]+b3[1]))/totmass
       posz=-(b0[nmer_count-1][2]*masses[0]+masses[1]*(b0[nmer_count-1][2]+b1[2])+masses[2]*(b0[nmer_count-1][2]+b1[2]+b2[2])+masses[3]*(b0[nmer_count-1][2]+b1[2]+b2[2]+b3[2]))/totmass  
       
       vcoords[0][:] = [posx,posy,posz]
       vcoords[1][:] = [vcoords[0][0]+b0[nmer_count-1][0],vcoords[0][1]+b0[nmer_count-1][1],vcoords[0][2]+b0[nmer_count-1][2]]   
       vcoords[2][:] = [vcoords[1][0]+b1[0],vcoords[1][1]+b1[1],vcoords[1][2]+b1[2]]  
       vcoords[3][:] = [vcoords[2][0]+b2[0],vcoords[2][1]+b2[1],vcoords[2][2]+b2[2]]  
       vcoords[4][:] = [vcoords[3][0]+b3[0],vcoords[3][1]+b3[1],vcoords[3][2]+b3[2]]  
         
     Coords[PartIndx][:] = [iosx+vcoords[1][0],iosy+vcoords[1][1],iosz+vcoords[1][2]]
     Coords[PartIndx+1][:] = [iosx+vcoords[2][0],iosy+vcoords[2][1],iosz+vcoords[2][2]]
     Coords[PartIndx+2][:] = [iosx+vcoords[3][0],iosy+vcoords[3][1],iosz+vcoords[3][2]]
     Coords[PartIndx+3][:] = [iosx+vcoords[4][0],iosy+vcoords[4][1],iosz+vcoords[4][2]]     

     return 


# Get the structure of each chain
def mer_identity(inp):  
  # Compute the zero-padding  
  shiftx = [i for i, rgb in enumerate(inp) if rgb[0] != 0 and rgb[1] != 0 and rgb[2] != 0][0]   
  identities = []
  names = []
  cis_names = ["CC","CDC","CDC","CC"]
  trans_names = ["CT","CDT","CDT","CT"]
  vinyl_names = ["CV2","CV1","CDV1","CDV2"]   
  f = 0
  # Find the sequence of CG types
  for i in range(inp.shape[0]):
      if (inp[i,5]==1 and (abs(inp[i,0])+abs(inp[i,1])+abs(inp[i,2])) != 0) :
          f += 1
          if f%dtseg_cis != 0: 
              continue
          else:    
              identities.append(int(0))
              names += cis_names
      elif (inp[i,4]==1 and (abs(inp[i,0])+abs(inp[i,1])+abs(inp[i,2])) != 0):  
          f += 1 
          if f%dtseg_trans != 0: 
              continue
          else:    
              identities.append(int(1))
              names += trans_names
      elif (inp[i,3]==1 and (abs(inp[i,0])+abs(inp[i,1])+abs(inp[i,2])) != 0):  
          f += 1 
          if f%dtseg_vinyl != 0: 
              continue
          else:
              identities.append(int(2))
              names += vinyl_names
  # Compute the number of monomers in this sample            
  img_nmer = len(identities)      
  return identities, shiftx, img_nmer, names

def ua_idx(shiftx,c,j):
    return shiftx + c + j

# Re-insert atomic detail to CG congigurations via the trained CNN
def create_chains(epoch):
 # Length of the simulation box   
 LXX,LYY,LZZ= 6.73965 , 6.73965 , 6.73965 
 hLXX,hLYY,hLZZ=LXX/2.0,LYY/2.0,LZZ/2.0
 
 # Counter for the test-frames index
 nframes = 0
 
 # Number of particles per chain
 chainlens = np.loadtxt('./chainlens.txt')
 save_path="./Data"+"_"+str(epoch)+"/"
 os.mkdir(save_path)
 pdimg=0
 chemistry = []
 chain_names = []
 nmer_count = 0
 PartIndx = 0
 pb0=np.zeros([nmer,3],dtype=np.float64)  
 tb0=np.zeros([nmer,3],dtype=np.float64)  
 for inp, tar in zip(test_input,test_target):
  inp = np.reshape(inp,(1,INPUT_WIDTH,6))   
  if len(chemistry) == 0:   
    Pcoords = np.zeros([int(chainlens[pdimg]),3],dtype=np.float64)
    Tcoords = np.zeros([int(chainlens[pdimg]),3],dtype=np.float64)
    if pdimg == 0:
     Pst = open(save_path+'PStart_'+str(com_list[nframes])+'.gro', 'w')
     Pst.write("PB_TCV_451045\n")
     Pst.write(str(npart)+"\n")
    
     pstd = open(save_path+"PStart_"+str(com_list[nframes])+'.dat', 'w')
     pstd.write("PB_TCV_451045\n")
     pstd.write(str(npart)+'\n')
     
     Tst = open(save_path+'TStart_'+str(com_list[nframes])+'.gro', 'w')
     Tst.write("PB_TCV_451045\n")
     Tst.write(str(npart)+"\n")
    
     tstd = open(save_path+"TStart_"+str(com_list[nframes])+'.dat', 'w')
     tstd.write("PB_TCV_451045\n")
     tstd.write(str(npart)+'\n')     

  identities, shiftx, img_nmer, names = mer_identity(inp[0])    
  chemistry += identities
  chain_names += names
  
  # Get a prediction of the trained model 
  pred = model.predict(inp)
  
  # Reverse the normalization 
  pred[0] += 1
  pred[0] /= 2.
  pred[0] *= 0.1711*2
  pred[0] += -0.1711  
  
  c = 0
 
  for ii in range(img_nmer):       
       if identities[ii] == 0: dtseg,merlen,masses = dtseg_cis,merlen_cis,masses_cis
       elif identities[ii] == 1: dtseg,merlen,masses = dtseg_trans,merlen_trans,masses_trans
       elif identities[ii] == 2: dtseg,merlen,masses = dtseg_vinyl,merlen_vinyl,masses_vinyl
       
       P_bv=np.zeros([dtseg,3],dtype=np.float64)
       T_bv=np.zeros([dtseg,3],dtype=np.float64)

       iosx=float(inp[0][c+shiftx+1,0])
       iosy=float(inp[0][c+shiftx+1,1])
       iosz=float(inp[0][c+shiftx+1,2])
       
       j=0 
       i = ua_idx(shiftx,c,j)
       P_bv[j]=[float(pred[0][i,0]),float(pred[0][i,1]),float(pred[0][i,2])] 
       T_bv[j]=[float(tar[i,0]),float(tar[i,1]),float(tar[i,2])] 

       j=1 
       i +=1
       P_bv[j]=[float(pred[0][i,0]),float(pred[0][i,1]),float(pred[0][i,2])] 
       T_bv[j]=[float(tar[i,0]),float(tar[i,1]),float(tar[i,2])] 
       
       j=2 
       i +=1
       P_bv[j]=[float(pred[0][i,0]),float(pred[0][i,1]),float(pred[0][i,2])] 
       T_bv[j]=[float(tar[i,0]),float(tar[i,1]),float(tar[i,2])] 
        
       j=3 
       i +=1
       P_bv[j]=[float(pred[0][i,0]),float(pred[0][i,1]),float(pred[0][i,2])] 
       T_bv[j]=[float(tar[i,0]),float(tar[i,1]),float(tar[i,2])] 
        
       c += dtseg  
       if(identities[ii]==0): 
              cis_d(P_bv,masses,iosx,iosy,iosz,pb0,nmer_count,Pcoords,chemistry,PartIndx)
              cis_d(T_bv,masses,iosx,iosy,iosz,tb0,nmer_count,Tcoords,chemistry,PartIndx)
              PartIndx += merlen 
       elif(identities[ii]==1): 
              trans_d(P_bv,masses,iosx,iosy,iosz,pb0,nmer_count,Pcoords,chemistry,PartIndx)
              trans_d(T_bv,masses,iosx,iosy,iosz,tb0,nmer_count,Tcoords,chemistry,PartIndx) 
              PartIndx += merlen 
       elif(identities[ii]==2): 
              vinyl_d(P_bv,masses,iosx,iosy,iosz,pb0,nmer_count,Pcoords,chemistry,PartIndx)
              vinyl_d(T_bv,masses,iosx,iosy,iosz,tb0,nmer_count,Tcoords,chemistry,PartIndx)
              PartIndx += merlen 
                 
       nmer_count += 1
  if len(chemistry) == nmer :
    for ii in range(Pcoords.shape[0]-1):
        while(Pcoords[ii+1][0]-Pcoords[ii][0]<-hLXX): Pcoords[ii+1][0]+=LXX
        while(Pcoords[ii+1][0]-Pcoords[ii][0]>hLXX): Pcoords[ii+1][0]-=LXX
        while(Pcoords[ii+1][1]-Pcoords[ii][1]<-hLYY): Pcoords[ii+1][1]+=LYY
        while(Pcoords[ii+1][1]-Pcoords[ii][1]>hLYY): Pcoords[ii+1][1]-=LYY
        while(Pcoords[ii+1][2]-Pcoords[ii][2]<-hLZZ): Pcoords[ii+1][2]+=LZZ
        while(Pcoords[ii+1][2]-Pcoords[ii][2]>hLZZ): Pcoords[ii+1][2]-=LZZ       
    for j in range(Pcoords.shape[0]):  
       Pst.write('%5d%5s%5s%5s%8.3f%8.3f%8.3f  0.0000  0.0000  0.0000\n'%((pdimg+1,'PB',chain_names[j],str(j+1+pdimg*chainlen)[-5:],Pcoords[j][0],Pcoords[j][1],Pcoords[j][2]))) 
       print(chain_names[j],Pcoords[j][0],Pcoords[j][1],Pcoords[j][2], file=pstd)   
       Tst.write('%5d%5s%5s%5s%8.3f%8.3f%8.3f  0.0000  0.0000  0.0000\n'%((pdimg+1,'PB',chain_names[j],str(j+1+pdimg*chainlen)[-5:],Tcoords[j][0],Tcoords[j][1],Tcoords[j][2]))) 
       print(chain_names[j],Tcoords[j][0],Tcoords[j][1],Tcoords[j][2], file=tstd)  
    pdimg+=1
    chemistry = []
    chain_names = []
    pb0=np.zeros([nmer,3],dtype=np.float64)
    tb0=np.zeros([nmer,3],dtype=np.float64)
    nmer_count = 0
    PartIndx = 0
   
    if (pdimg%nchain) == 0:  
        nframes+=1
        pdimg=0
        print(nframes)
        Pst.write('%10.5f%10.5f%10.5f\n'%((LXX , LYY , LZZ)))
        Pst.close
        pstd.close
        Tst.write('%10.5f%10.5f%10.5f\n'%((LXX , LYY , LZZ)))
        Tst.close
        tstd.close
  
            
# Get predictions for the test set, for the epochs given below
epoch_list = ["0001","0050","0100","0200","0300","0400","0500","0600","0700","0800","0900","1000"]
epochs = [1,50,100,200,300,400,500,600,700,800,900,1000]
counter = 0
for i in epoch_list:   
        checkpoint_path = "./tmp/cp-"+ str(i)+ ".ckpt"
        model.load_weights(checkpoint_path)
        create_chains(epochs[counter])
        counter += 1

