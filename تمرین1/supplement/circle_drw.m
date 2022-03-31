function ret = circle_drw(img_mnpl,center_x_ratio,center_y_ratio,radius_max_ratio,radius_thickness)
img_mnpl_size = size (img_mnpl);

%center_x_ratio = 0.45;
%center_y_ratio = 0.45;
%radius_max_ratio = 0.95;
%radius_thickness = 0.5;

radius = min(min(center_x_ratio.*img_mnpl_size(2),center_y_ratio.*img_mnpl_size(1)),min((1-center_x_ratio).*img_mnpl_size(2),(1-center_y_ratio).*img_mnpl_size(1)));

radius = radius .* radius_max_ratio;
center_x = center_x_ratio.*img_mnpl_size(2);
center_y = center_y_ratio.*img_mnpl_size(1);

 
for i= 1:1:img_mnpl_size(1)
     for j= 1:1:img_mnpl_size(2)
        if (abs(sqrt((i-center_y).^2+(j-center_x).^2) - radius) < radius_thickness)
           img_mnpl(i,j) = 255;
        end
	end
end

ret = img_mnpl;
end