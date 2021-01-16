import numpy as np
from scipy.ndimage import maximum_filter as max_filter

def norm_pdf(x,mean,sigma):
    return (1/(np.sqrt(2*np.pi)*sigma))*(np.exp(-0.5*(((x-mean)/sigma)**2)))

# Model parameters:
# N, number of samples to look back
# th, probability threshold
class KDE_subtractor:

    def __init__(self, N, th, img_shape):
        H, W, C = img_shape

        self.N = N
        self.th = th
        self.samples = np.zeros((N, H, W, C)) # array to store samples
        self.sample_index = 0
        pass

    def filter_frame(self, frame, th2=None, k_shape=None):
        # Add sample
        self.samples[self.sample_index, :,:,:] = frame
        self.sample_index += 1
        if(self.sample_index >= self.N):
            self.sample_index = 0

        # Get variances
        shifted_samples = np.zeros_like(self.samples)
        shifted_samples[:-1,:,:,:] = self.samples[1:,:,:,:]
        shifted_samples[0,:,:,:] = self.samples[-1,:,:,:]
        sigma = np.abs(self.samples-shifted_samples)/(0.68*1.41421356237)

        # Calculating probabilities
        gauss = norm_pdf(frame, self.samples, sigma)
        probability = np.sum(np.prod(gauss, axis=-1), axis=0)/self.N

        fg_mask = probability < self.th

        # Suppressing false detection
        if th2 != None:
            fg_index = np.where(fg_mask) # Finding indicies of all foreground pixels

            prob_N = max_filter(probability, size=k_shape, mode='constant')

            prod_prob = np.where(fg_mask, prob_N, 1)
            prob_C = np.zeros_like(frame)
            fg_ind_arr = np.array(fg_mask)
            Hk, Wk = k_shape
            for i in range(fg_ind_arr.shape[1]):
                center_y, center_x = fg_ind_arr[0,i], fg_ind_arr[1,i]
                y = np.arange(center_y-Hk//2, center_y+Hk//2+1)
                x = np.arange(center_x-Wk//2, center_x+Wk//2+1)
                
                prob_C = np.prod(prod_prob[y,x])

            change = np.logical_and(prob_N > self.th, prob_C > th2)
            # Converting some foreground pixels back to background
            fg_mask = np.where(np.logical_and(change, fg_mask), 0, fg_mask)
            return fg_mask

        else:
            return fg_mask