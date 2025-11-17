classdef gacelleFFT 
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% 
% This class contains GACELLE compatible FFT related functions
%
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
% Date created: 17 July 2025
% Date modified:
%

    properties (Constant)
        epsilon = 1e-8;
    end

    methods(Static)

        % Apply 1D FFT to the specified dimension
        function output = fft(input,dim)
        % 
        % Input
        % ---------------
        % input         : N-D array (N>=2) 
        % 
        % Output
        % ---------------
        % output         :2D FFT of input
        %
        % Description: 
        %

            if nargin < 2
                dim = 1;
            end
        
            % normalisation factor
            fctr    = size(input,dim);
            % FFT
            output  = fftshift(fft(ifftshift(input,dim),[],dim),dim)/ sqrt(fctr);
        
        end

        % Apply 2D FFT on the first 2 dimensions
        function output = fft2(input)
        % 
        % Input
        % ---------------
        % input         : N-D array (N>=2) 
        % 
        % Output
        % ---------------
        % output         :2D FFT of input
        %
        % Description: 
        %
        
            % normalisation factor
            fctr    = prod(size(input,1:2));
            % FFT
            output  = fftshift(fft(ifftshift(input,1),[],1),1);
            % FFT
            output  = fftshift(fft(ifftshift(output,2),[],2),2) / sqrt(fctr);
        
        end

        % Apply 3D FFT on the first 3 dimensions
        function output = fft3(input)
        % 
        % Input
        % ---------------
        % input         : N-D array (N>=2) 
        % 
        % Output
        % ---------------
        % output         :2D FFT of input
        %
        % Description: 
        %
        
            % normalisation factor
            fctr    = prod(size(input,1:3));
            % FFT
            output  = fftshift(fft(ifftshift(input,1),[],1),1);
            % FFT
            output  = fftshift(fft(ifftshift(output,2),[],2),2);
            % FFT
            output  = fftshift(fft(ifftshift(output,3),[],3),3) / sqrt(fctr);
        
        end

        % Apply m-DFFT on the first m dimensions of an N-D array
        function output = fftm(input,m)
        % 
        % Input
        % ---------------
        % input         : N-D array (N>=m) 
        % m             : number of dimwnsion that FFT to be applied
        % 
        % Output
        % ---------------
        % output         :mD FFT of input
        %
        % Description: 
        %
        
            % normalisation factor
            fctr    = prod(size(input,1:m));
            output  = input;
            for k = 1:m
                % FFT
                output = fftshift(fft(ifftshift(input,m),[],m),m);
            end
            % FFT
            output = output / sqrt(fctr);
        
        end

        % Apply 1D inverse FFT to the specified dimension
        function output = ifft(input,dim)
        % 
        % Input
        % ---------------
        % input         : N-D array (N>=1) 
        % 
        % Output
        % ---------------
        % output         :1D FFT of input
        %
        % Description: 
        %

            if nargin < 2
                dim = 1;
            end
        
            % normalisation factor
            fctr    = size(input,dim);
            % iFFT
            output  = fftshift(ifft(ifftshift(input,dim),[],dim),dim) * sqrt(fctr);
        
        end

        % Apply 2D inverse FFT on the first 2 dimensions
        function output = ifft2(input)
        % 
        % Input
        % ---------------
        % input         : N-D array (N>=2) 
        % 
        % Output
        % ---------------
        % output         :2D FFT of input
        %
        % Description: 
        %
        
            % normalisation factor
            fctr    = prod(size(input,1:2));
            % FFT
            output  = fftshift(ifft(ifftshift(input,1),[],1),1);
            % FFT
            output = fftshift(ifft(ifftshift(output,2),[],2),2) * sqrt(fctr);
        
        end

        % Apply 3D inverse FFT on the first 3 dimensions
        function output = ifft3(input)
        % 
        % Input
        % ---------------
        % input         : N-D array (N>=3) 
        % 
        % Output
        % ---------------
        % output         :3D FFT of input
        %
        % Description: 
        %
        
            % normalisation factor
            fctr    = prod(size(input,1:2));
            % FFT
            output  = fftshift(ifft(ifftshift(input,1),[],1),1);
            % FFT
            output  = fftshift(ifft(ifftshift(output,2),[],2),2);
            % FFT
            output  = fftshift(ifft(ifftshift(output,3),[],3),3) * sqrt(fctr);
        
        end

        % Apply m-D iFFT on the first m dimensions of an N-D array
        function output = ifftm(input,m)
        % 
        % Input
        % ---------------
        % input         : N-D array (N>=m) 
        % m             : number of dimwnsion that FFT to be applied
        % 
        % Output
        % ---------------
        % output         :mD FFT of input
        %
        % Description: 
        %
        
            % normalisation factor
            fctr    = prod(size(input,1:m));
            output  = input;
            for k = 1:m
                % FFT
                output = fftshift(ifft(ifftshift(input,m),[],m),m);
            end
            % FFT
            output = output * sqrt(fctr);
        
        end

    end
end