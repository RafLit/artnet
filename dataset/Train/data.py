import numpy as np
import cv2
from PIL import Image
all_data = []
ds = 'Test'
for nr in range(1,663):
  nrs = str(nr)
  nrs0 = (4-len(nrs))*'0'+nrs
  print(nr)


  #image pos
  p = ds + '_Positive/'+ds+'_Pos_sample_'+nrs0+'.png'
  pos = np.asarray(Image.open(p).convert('RGB').resize(size=(192,256)))/255.
  #mask
  p = ds + '_Positive_Tool_Mask/'+ds+'_Pos_sample_'+nrs0+'_Tool_Mask.png'
  mat = np.asarray(Image.open(p).convert('L'))/255.
  maskpos= np.resize(mat, (192, 256))
  #edge
  p1 = ds + '_Positive_EdgeLine_1/'+ds+'_Pos_sample_'+nrs0+'_EdgeLine_1.png'
  p2 = ds + '_Positive_EdgeLine_2/'+ds+'_Pos_sample_'+nrs0+'_EdgeLine_2.png'
  mat1 = np.asarray(Image.open(p).convert('L'))/255.
  mat2 = np.asarray(Image.open(p).convert('L'))/255.
  resized1 = np.resize(mat1, (192, 256))
  resized2 = np.resize(mat2, (192, 256))
  edgepos = np.logical_or(resized1, resized2)
  #mid
  p = ds + '_Positive_MidLine/'+ds+'_Pos_sample_'+nrs0+'_MidLine.png'
  mat = np.asarray(Image.open(p).convert('L'))/255.
  midpos = np.resize(mat, (192, 256))
  #tip
  p = ds + '_Positive_TipPoint/'+ds+'_Pos_sample_'+nrs0+'_TipPoint.png'
  mat = np.asarray(Image.open(p).convert('L'))/255.
  tippos = np.resize(mat, (192, 256))

  #pos
  pos = [pos, maskpos, edgepos, midpos, tippos, True]
  all_data.append(pos)
  #image neg
  p = ds + '_Negative/'+ds+'_Neg_sample_'+nrs0+'.png'
  neg = np.asarray(Image.open(p).convert('RGB').resize(size=(192,256)))/255.

  maskneg = np.zeros_like(maskpos)
  edgeneg = np.zeros_like(edgepos)
  midneg = np.zeros_like(midpos)
  tipneg = np.zeros_like(tippos)
  neg = [neg, maskneg, edgeneg, midneg, tipneg, False]
  all_data.append(neg)
  
ad = np.array(all_data, dtype=object)  
np.save('testing_data', ad)

  
#mat = cv2.imread('Train_Positive_Tool_Mask/Train_Pos_sample_0662_Tool_Mask.png', cv2.IMREAD_UNCHANGED)
#print(mat.shape)
#print(mat)
