clc;
clear;
close all;

% dbstop if error;

load 'nyu_names.mat';
load 'nyu_train_images.mat';
load 'nyu_train_labels.mat';

for idx = 1:size(imgs, 4)
    img = imgs(:, :, :, idx);
    lab = labs(:, :, idx);
    imgname = strcat(num2str(idx), '.jpg');
    
    % get the structure from the given images and label and imgname;
    struct_out = create_struct(img, lab, names, imgname);
    
    % convert to xml;
    xmlname =  strcat('Annotation\', num2str(idx), '.xml');
    out = struct2xml(struct_out);
    out(20:36) = '';
    xmlwrite(out, xmlname);
    % Now save both xml and image in separate folder;
    imwrite(img, ['Data\', imgname], 'jpg');
    
    
    disp([imgname, ' ', xmlname]);
end