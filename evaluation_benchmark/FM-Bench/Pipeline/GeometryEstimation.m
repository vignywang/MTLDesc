function GeometryEstimation(wkdir, dataset, ratio, name)
% Matching descriptors and save results
disp('Running FM estimation...');

dataset_dir = [wkdir 'Dataset/' dataset '/'];
matches_root = [wkdir 'Matches/' dataset '/'];
matches_dir = [matches_root name '_'];

results_root = [wkdir 'Results/' dataset '/'];
if exist(results_root, 'dir') == 0
    mkdir(results_root)
end

results_dir = [results_root name '_'];

pairs_gts = dlmread([dataset_dir 'pairs_with_gt.txt']);
pairs_which_dataset = importdata([dataset_dir 'pairs_which_dataset.txt']);

pairs = pairs_gts(:,1:2);
l_pairs = pairs(:,1);
r_pairs = pairs(:,2);
F_gts = pairs_gts(:,3:11);

load([matches_dir sprintf('%.2f', ratio) '.mat']);
Results = Matches;
num_pairs = size(pairs,1);
disp(sprintf('There are total %d pairs', num_pairs));

for idx = 1 : num_pairs
    disp(sprintf('Computing %d pair...', idx));
    l = l_pairs(idx);
    r = r_pairs(idx);
    
    Results{idx}.dataset = dataset;
    Results{idx}.subset = pairs_which_dataset{idx};
    Results{idx}.l = l;
    Results{idx}.r = r;
    Results{idx}.F_gt = reshape(F_gts(idx,:), 3, 3)';
    
    X_l = Results{idx}.X_l;
    X_r = Results{idx}.X_r;
    
    F_hat = [];
    inliers = [];
    status = 3; % 0 stands for good, others are bad estimations.
    
    try
        [F_hat, inliers, status] = estimateFundamentalMatrix(X_l, X_r, 'Method','RANSAC', 'NumTrials', 2000);
    catch
        disp('Estimation Crash');
    end
    
    Results{idx}.F_hat = F_hat;
    Results{idx}.inliers = inliers;
    Results{idx}.status = status;
end

results_file = [results_dir sprintf('%.2f', ratio) '.mat'];
save(results_file, 'Results');

disp('Finished.');
end