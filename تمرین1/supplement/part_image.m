function ret = part_image(img,mag_ratio,sel_ratio)
%img = imread('cameraman.tif');
%mag_ratio = 0.2;
%sel_ratio =0.75;

img_size = size(img);

 % memory reservation for manipulated image
img_mnpl = zeros(ceil(img_size(1).*(1+mag_ratio)),ceil(img_size(2).*(1+mag_ratio)));
img_mnpl = im2uint8(img_mnpl);

 for i= 1:1:img_size(1)
     for j= 1:1:img_size(2)
	     if i < ceil((sel_ratio).*img_size(1)) &&  j < ceil((sel_ratio).*img_size(2))
		    img_mnpl(i,j) = img(i,j);
		 else
		    img_mnpl(i+floor(img_size(1).*(mag_ratio)),j+floor(img_size(2).*(mag_ratio))) = img(i,j);
         end
	end
 end
 
ret = img_mnpl;
end