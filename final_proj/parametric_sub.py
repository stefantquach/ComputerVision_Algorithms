import numpy as np
# from scipy.stats import multivariate_normal

def norm_pdf(x,mean,sigma):
    return (1/(np.sqrt(2*np.pi)*sigma))*(np.exp(-0.5*(((x-mean)/sigma)**2)))

# Parameters needed for construction:
# alpha:  Learning rate
# T:      proportion of data to be accounted to background

class mixed_gaussian_subtractor:
    def __init__(self, K, T, alpha, first_frame):
        # T ranges from 0-1
        # K is a small integer (3-5)
        # alpha is learning rate (0-1)
        # C is number of channels on pixel
        H,W = first_frame.shape

        # initialize arrays of parameters
        self.K = K
        self.T = T
        self.alpha = alpha

        self.w = np.zeros((K,H,W))/K # weights

        self.avg = np.ones((K,H,W))*122 # means
        self.variance = np.ones((K, H, W))*400 # std deviations


    def filter_frame(self, frame):
        H,W = frame.shape

        # In case variance gets too small.
        self.variance[0][np.where(self.variance[0] < 1)] = 10
        self.variance[1][np.where(self.variance[1] < 1)] = 5
        self.variance[2][np.where(self.variance[2] < 1)] = 1
        

        ################# Updating model #########################
        # Finding matched points
        low_lim = self.avg - 2.5*np.sqrt(self.variance)
        upp_lim = self.avg + 2.5*np.sqrt(self.variance)
        # (K, H, W) logical array. True is matched, False otherwise
        mask = np.logical_and(frame > low_lim, frame < upp_lim)
        # print("Mask shape", mask.shape)
        matched_ind = np.where(mask)
        K_ind, H_ind, W_ind = matched_ind
        
        background = np.any(mask, axis=0) # array of pixels that match at least one distribution

        # updating weights
        self.w = (1-self.alpha)*self.w + self.alpha*mask

        # updating avg and variance
        for i in range(self.K):
            ind = np.where(K_ind == i)
            K_, H_, W_ = K_ind[ind], H_ind[ind], W_ind[ind]
            matched = (K_, H_, W_)
            pixels = (H_, W_)
            # print(self.variance[matched].shape, frame[pixels].shape)

            p = self.alpha*norm_pdf(frame[pixels], self.avg[matched], np.sqrt(self.variance[matched]))
            self.avg[matched] = (1-p)*self.avg[matched] + p*frame[pixels]
            self.variance[matched] = (1-p)*self.variance[matched] + p*(frame[pixels]-self.avg[matched])**2
 
        # updating least probable gaussians for non-matched pixels
        non_matched = np.logical_not(background)
        self.avg[0][non_matched] = frame[non_matched]
        self.variance[0][non_matched] = 200 # Just some high variance
        self.w[0][non_matched] = 0.1

        # Normalizing w
        sum = np.sum(self.w, axis=0)
        self.w = self.w/sum

        # # Reordering w
        # ind = np.argsort(self.w/(self.variance), axis=0)
        # # print(ind.shape)
        # self.w = np.take_along_axis(self.w, ind, axis=0)
        # # print(self.w.shape)
        # self.avg = np.take_along_axis(self.w, ind, axis=0)
        # self.variance = np.take_along_axis(self.w, ind, axis=0)
        # # print(self.w)

        ################# Calculating background #####################
        # Calculating B TODO Fix this later
        # B = np.argmin(np.where(np.cumsum(self.w, axis=0)>self.T), axis=0)
        # print((B!=0).any())

        fg_mask = np.logical_or(frame <= low_lim, frame >= upp_lim)
        # print(fg_mask.shape)
        # print(mask[1:,:,:].shape)
        return np.any(mask[:2,:,:], axis=0)

    def filter_frame2(self, frame):
        pass