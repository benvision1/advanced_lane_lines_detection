import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        self.line_pixels_x = None
        self.line_pixels_y = None
        
        self.poly_fit = None
        self.line_fit_x = None
        
        self.accumilated_poly_fit = []
        self.accumilated_line_fit_x = []
        
        self.best_line_fit_x = None
        self.best_poly_fit = None
        
        self.curvature_rad = None
        self.img_size = None
        self.covariance = None
        self.base_x = None
        self.ploty = None
    
    def reset(self):
        self.line_pixels_x = None
        self.line_pixels_y = None
        
        self.poly_fit = None
        self.line_fit_x = None
        
        self.accumilated_poly_fit = []
        self.accumilated_line_fit_x = []
        
        self.best_line_fit_x = None
        self.best_poly_fit = None
        
        self.curvature_rad = None
        self.covariance = None
        self.base_x = None
        
        
    def fit_polynomial(self, fitx, fity):
        self.line_pixels_x = fitx
        self.line_pixels_y = fity
        self.ploty = np.linspace(0, self.img_size[1] - 1, self.img_size[1])
        self.poly_fit = np.array(np.polyfit(self.line_pixels_y, self.line_pixels_x, 2))
        self.line_fit_x = np.array(self.poly_fit[0]*self.ploty**2 + self.poly_fit[1]*self.ploty + self.poly_fit[2])
        
        self.base_x  = self.poly_fit[0]*self.img_size[1]**2 + self.poly_fit[1]*self.img_size[1] + self.poly_fit[2]
        self.update_line()
    
    
    def update_accumilated_fits(self):
        self.accumilated_poly_fit.append(self.poly_fit)
        self.accumilated_line_fit_x.append(self.line_fit_x)
    
    def update_bestpos(self):
        self.best_poly_fit = np.mean(self.accumilated_poly_fit, axis=0)
        self.best_line_fit_x = np.mean(self.accumilated_line_fit_x, axis=0)
    
    def update_curvature(self):
        ym_per_pix = 30.0/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(self.line_pixels_y)
        # print('line pixel y {}'.format(self.line_pixels_y.shape))
        # print('line pixel x {}'.format(self.line_pixels_x.shape))
        
        fit_cr, cov = np.polyfit(self.line_pixels_y* np.array([ym_per_pix]), self.line_pixels_x*np.array([xm_per_pix]), 2, cov=True)
        self.covariance = np.diag(cov)
        # self.curvature_rad = ((1 + (2*fit_cr[0]*(self.img_size[0] - 1)*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        self.curvature_rad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    
    def update_line(self):
        self.update_accumilated_fits()
        self.update_bestpos()
        self.update_curvature()
        