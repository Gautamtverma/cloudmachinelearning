function struct_out = create_struct(image, label, names, imgfilename)


struct_out = [];


[ht, wd, k] = size(image);
vals = unique(label);

struct_out.annotation.folder = 'Annotation';
struct_out.annotation.filename = imgfilename;
struct_out.annotation.path = ['/', imgfilename];
struct_out.annotation.size.width = wd;
struct_out.annotation.size.height = ht;
struct_out.annotation.size.depth = 3;

% struct_out.annotation.segmented = 0;

indx = 1;
for id = 1:length(vals)
    if (vals(id) ~= 0)
        label_name = names{vals(id)};
        
        [y, x] = find(label == vals(id));
        
        minx = min(x);
        miny = min(y);
        maxx = max(x);
        maxy = max(y);
        
        struct_out.annotation.object{indx}.name = label_name;
        struct_out.annotation.object{indx}.pose = 'Unspecified';
        struct_out.annotation.object{indx}.bndbox.xmin = minx;
        struct_out.annotation.object{indx}.bndbox.ymin = miny;
        struct_out.annotation.object{indx}.bndbox.xmax = maxx;
        struct_out.annotation.object{indx}.bndbox.ymax = maxy;
        
        indx = indx+1;
    end
end
