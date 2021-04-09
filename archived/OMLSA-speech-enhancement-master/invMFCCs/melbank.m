% ==========================================================================
%  [mel_filter,fc,a] = melbank(fs,NumLinFilts,NumLogFilts,frame_length,lin_limit)
% 
%  Description: This function computes the mel filters, including the two 
%               half filters at the beginning and end.
% 
%  Input Arguments:
% 	Name: fs
% 	Type: scalar
% 	Description: sampling rate
% 
% 	Name: NumLinFilts
% 	Type: scalar
% 	Description: number of linearly spaced filters
% 
% 	Name: NumLogFilts
% 	Type: scalar
% 	Description: number of logarithmicaly spaced filters
% 
% 	Name: frame_length
% 	Type: scalar
% 	Description: window length
% 
% 	Name: lin_limit (optional)
% 	Type: scalar (default 1000)
% 	Description: frequency that divides the lin and log spaced filters 
% 
%  Output Arguments:
% 	Name: mel_filter
% 	Type: matric 
% 	Description: a mel filter bank
% 
% 	Name: fc
% 	Type: vector 
% 	Description: center frequencies of the filters
% 
% 	Name: a
% 	Type: vector 
% 	Description: amplitudes of the filters to assure equal area
% 
% 
%  Reference:
%  [1] R. Vergin and A. Farhat, "Generalized Mel Frequency Cepstral 
%   Coefficients for Large-Vocabulary Speaker-Independent Continuous-Speech
%   Recognition," IEEE Transactions on Speech and Audio Processing, vol. 7,
%   no. 5, Sep 1999, pp. 525-532.
% 
% --------------------------------------------------------------------------
%  Notes:
%  The amplitudes in a are already included in the weight matrix mel_filter.  If a
%  filter weighting is desired with common amplitude between all the 
%  filters, the vector a should be used to rescale the weight matrix w.
% 
%  The computations leading to the equations for mel_filter are included in Laura's
%  notebook for 02/06/08.
% 
% --------------------------------------------------------------------------
%  Author: Laura E. Boucheron
% 
%  Creation Date: 06 February 2008
% 
%  Copyright 2007, Board of Regents at New Mexico State University,
%  All Rights Reserved
% --------------------------------------------------------------------------
%  Revision History:
%  Adapted from the original mel_filter code 06/04/08.
%  Added variable Linear/Log boundy 2 April 09. Steven Sandoval
%  Major function call changes - June 2011, Steven Sandoval
% ==========================================================================
%
