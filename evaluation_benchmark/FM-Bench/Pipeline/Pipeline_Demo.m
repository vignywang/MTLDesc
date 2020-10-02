clear; clc; close all;
wkdir = '../'; % The root foler of FM-Bench
% addpath([wkdir 'vlfeat-0.9.21/toolbox/']);
%vl_setup;

Datasets = {'TUM', 'KITTI', 'Tanks_and_Temples', 'CPC'};
% Datasets = {'KITTI'};

% ratio=0.9;
% ratio=1.0;
ratio=0.8;

% names={'mlifeat128megacoco09', 'mlifeat128megacoco08', 'mlifeat128megacoco07', 'mlifeat128megacoco06', 'mlifeat128megacoco05', 'mlifeat128megacoco04', 'mlifeat128megacoco03', 'mlifeat128megacoco02', 'mlifeat128megacoco01'};
% names={'mlifeat128megacoco08', 'mlifeat128megacoco07', 'mlifeat128megacoco05', 'mlifeat128megacoco04', 'mlifeat128megacoco02', 'mlifeat128megacoco01'};
% names={'superpointselftrained30'};
% names={'superpointselftrained15'};
% names={'superpointselftrained30', 'mlifeat128megacoco08', 'mlifeat128megacoco04'};
% names={'aslfeat'};
names={'dspsift'};
% name='mlifeat128megacoco09';
% name='mlifeat128megacoco08';
%name='aslfeat_ms';

for n = 1 : length(names)
    name = names{n};
    disp(sprintf('Method: %s', name))
    for s = 1 : length(Datasets)
        disp(sprintf('Evaluating %s ...', Datasets{s}))
        dataset = Datasets{s};
    
        % An example for DoG detector
        % FeatureDetection(wkdir, dataset, name);

        % An example for SIFT descriptor
        % FeatureExtraction(wkdir, dataset, name);
   
        % An example for exhaustive nearest neighbor matching with ratio test
        FeatureMatching(wkdir, dataset, ratio, name);
    
        % An example for RANSAC based FM estimation
        GeometryEstimation(wkdir, dataset, ratio, name);
    
    end
end


