function FeatureDetection(wkdir, dataset, name)
% Detect and save DoG keypoints

disp('Detecting keypoints...');

dataset_dir = [wkdir 'Dataset/' dataset '/'];

feature_root = [wkdir 'Features/' dataset '/'];
if exist(feature_root, 'dir') == 0
    assert(false)
    % mkdir(feature_dir);
end
feature_dir = [feature_root name '_'];

pairs_gts = dlmread([dataset_dir 'pairs_with_gt.txt']);
pairs_which_dataset = importdata([dataset_dir 'pairs_which_dataset.txt']);

pairs = pairs_gts(:,1:2);
l_pairs = pairs(:,1);
r_pairs = pairs(:,2);

num_pairs = size(pairs,1);
for idx = 1 : num_pairs
    l = l_pairs(idx);
    r = r_pairs(idx);
    
    I1 = imread([dataset_dir pairs_which_dataset{idx} 'Images/' sprintf('%.8d.jpg', l)]);
    I2 = imread([dataset_dir pairs_which_dataset{idx} 'Images/' sprintf('%.8d.jpg', r)]);
    
    if size(I1,3) == 3
        I1gray = rgb2gray(I1);
    else
        I1gray = I1;
    end
    
    if size(I2,3) == 3
        I2gray = rgb2gray(I2);
    else
        I2gray = I2;
    end
    
    path_l = [feature_dir sprintf('%.4d_l', idx)];
    path_r = [feature_dir sprintf('%.4d_r', idx)];
    
    if exist([path_l '.keypoints'], 'file') == 2 && exist([path_r '.keypoints'], 'file') == 2
        continue;
    else
        disp(sprintf('There is no %s', [path_l '.keypoints']));
        disp(sprintf('There is no %s', [path_r '.keypoints']));
        assert(false);
    end
    
    % keypoints_l = vl_sift(single(I1gray))';
    % write_keypoints([path_l '.keypoints'], keypoints_l);
    
    % keypoints_r = vl_sift(single(I2gray))';
    % write_keypoints([path_r '.keypoints'], keypoints_r);
end

disp('Finished.');
end

