import numpy as np
from scipy.interpolate import interp1d
from numpy.random import uniform
'''
% sampleContour
% Modified from Bob Hou's code. -- K.Ehinger, 2018
% Modified from K.Ehinger, 2018
% Input:
% im = silhouette image ("figure" regions should be 1 and "ground" regions
%    should be 0)
% ntan = number of points to sample around the boundary contour
% multiflag = if set to 1, the code will use all "figure" objects over a
%    size threshold; if set to 0 only the longest "figure" contour will be
%    used
% lengthTh = minimum contour length (in pixels) when allowing multiple
%    objects per image (this input is ignored when multiflag is off -- the
%    longest available contour will always be used, even if it is smaller
%    than this threshold)
%
% Output:
% if multiflag == 0, bi is a nTan x 2 double array
% if multiflag == 1, bi is a cell array of nTan x 2 doubles

'''
def sampleContour(pointData, ntan):
    b = pointData[:, :2] / pointData[:, 2, np.newaxis]

    s = np.concatenate(([0], np.cumsum(np.sqrt(np.sum(np.diff(b, axis=0)**2, axis=1)))))
    _, ind = np.unique(s, return_index=True)
    duplicate_ind = np.setdiff1d(np.arange(s.shape[0]), ind)
    s = np.delete(s, duplicate_ind)
    pointData = np.delete(pointData, duplicate_ind, axis=0)

    arclen = s[-1] # total arclength of boundary
    tanspace = arclen / ntan # arclength distance between adjacent tangents

    # arc length positions of sampled points
    si = np.mod(uniform(0, arclen) + tanspace * np.arange(ntan), arclen)

    # (x,y) locations of sampled points in arclength order
    bi = interp1d(s, pointData, axis=0)(si)

    return bi