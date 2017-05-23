# Copyright 2017 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy import optimize

class GaussianFitError(RuntimeError): pass

def gaussian2D(height, center_x, center_y, width_x, width_y, rotation):
    width_x = float(width_x)
    width_y = float(width_y)
    
    rotation = np.deg2rad(rotation)
    center_xp = center_x*np.cos(rotation) - center_y*np.sin(rotation)
    center_yp = center_x*np.sin(rotation) + center_y*np.cos(rotation)
    
    def rotgauss(x,y):
        xp = x*np.cos(rotation) - y*np.sin(rotation)
        yp = x*np.sin(rotation) + y*np.cos(rotation)
        g = height*np.exp(-((center_xp-xp)/width_x)**2/2.0 - ((center_yp-yp)/width_y)**2/2.)
        return g
    return rotgauss
            
# def moments(data):
#     #uses location of peak as seed for center
#     # total = data.sum()
#     # Y,X = np.indices(data.shape)
#     x = (np.where(data==data.max())[1][0]).astype(np.float)        #(X*data).sum()/total # BUG HERE USES MAX, what happens if tie?
#     y = (np.where(data==data.max())[0][0]).astype(np.float)        #(Y*data).sum()/total
#
#     print x, np.where(data==data.max())[1]
#     print y, np.where(data==data.max())[0]
#
#
#     col = data[:, int(x)]
#     width_x = np.sqrt(abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
#
#     print abs((np.arange(col.size)-x)**2*col).sum()/col.sum()
#
#     print 'moments', x, width_x, 0.0
#     import sys
#     sys.exit()
#     return height, y, x, width_y, width_x, 0.0
#
#
#     row = data[int(y), :]
#     width_y = np.sqrt(abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
#     height = data.max()
#     #


def moments2(data):
    #uses original method from website for finding center
    total = data.sum()
    Y,X = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(x)]
    width_x = np.sqrt(abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(y), :]
    width_y = np.sqrt(abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    # print height, y, x, width_y, width_x, 0.0
    #
    # import sys
    # sys.exit()

    return height, y, x, width_y, width_x, None
    
def fitgaussian2D(data):
    params = moments2(data)
    def errorfunction(p):
        p2 = np.array([p[0], params[1], params[2], np.abs(p[1]), np.abs(p[2]), p[3]])


        val = np.ravel(gaussian2D(*p2)(*np.indices(data.shape)) - data)

        return (val**2).sum()

    res = optimize.minimize(errorfunction, [ params[0], params[3], params[4], 0.0 ], method='Nelder-Mead', options={'maxfev':2500})
    p = res.x
    p2 = np.array([p[0], params[1], params[2], np.abs(p[1]), np.abs(p[2]), p[3]])
    success = res.success
    if not success and res.status != 2: # Status 2 is loss of precision; might need to handle this separately instead of passing...
        print success
        print res.message
        print res.status
        raise GaussianFitError('Gaussian optimization failed to converge:\n%s' % res.message)

    return p2

def fitgaussian2D_fixedcenter(data):
    params = moments2(data)
    if any(np.isnan(params)):
        params = moments2(data)
    fixedcenter_x = params[2]
    fixedcenter_y = params[1]
    new_params = tuple(params[a] for a in [0,3,4,5])

    def gaussian2D_fixedcenter(height, width_x, width_y, rotation):
        width_x = float(width_x)
        width_y = float(width_y)
        
        rotation = np.deg2rad(rotation)
        center_xp = fixedcenter_x*np.cos(rotation) - fixedcenter_y*np.sin(rotation)
        center_yp = fixedcenter_x*np.sin(rotation) + fixedcenter_y*np.cos(rotation)
        
        def rotgauss(x,y):
            xp = x*np.cos(rotation) - y*np.sin(rotation)
            yp = x*np.sin(rotation) + y*np.cos(rotation)
            g = height*np.exp(-((center_xp-xp)/width_x)**2/2.0 - ((center_yp-yp)/width_y)**2/2.)
            return g
        return rotgauss    
    
    errorfunction = lambda p: np.ravel(gaussian2D_fixedcenter(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, new_params)
    new_p = list((p[0], fixedcenter_y, fixedcenter_x, p[1],p[2],p[3]))
    return params, new_p
