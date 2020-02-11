import numpy as np

def mse(img_mov,img_ref):
    mse = np.mean((img_mov - img_ref) ** 2)
    return mse

def mae(img_mov,img_ref):
    mae = np.mean(np.sum(abs(img_mov - img_ref)))
    return mae


#sum of squared errors
def sse(img_mov, img_ref):
    img1 = img_mov.astype('float64')
    img2 = img_ref.astype('float64')
    r = (img1 - img2)**2
    sse = np.sum(r.ravel())
    sse /= r.ravel().shape[0]
    return sse

def mutual_information(img_mov,img_ref):
    hgram, x_edges, y_edges = np.histogram2d(img_mov.ravel(), img_ref.ravel(), bins=20)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))