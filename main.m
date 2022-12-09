clc,clear;
%%
formatSpec = '%f 1:%f 2:%f 3:%f 4:%f 5:%f 6:%f 7:%f 8:%f';
classes = ["chair", "desk", "sofa", "table", "toilet"];
c = size(classes,2);    % #classes
%% inputs for PCA, PNN and KNN
d_p = 61; % reduced dimension
start= -4;
stop = 0;
sigma_pnn = logspace(start,stop,stop-start+1);
Knn = [1 5];
%% read datasets
D.total = []; % learning set
T.total = []; % test set
D.targets = [];
T.targets = [];
for i = 1:c
    temp = readmatrix(fullfile(pwd,classes(i),"train.txt"));
    D.total = [D.total;temp];
    D.targets = [D.targets;i*ones(size(temp,1),1)];
    temp = readmatrix(fullfile(pwd,classes(i),"test.txt"));
    T.total = [T.total;temp];
    T.targets = [T.targets;i*ones(size(temp,1),1)];
end
Uclasses = unique(D.targets);
%% principle component analysis (PCA)
[D.total_pca, D.targets, UW, mu, W] = PCA(D.total', D.targets, d_p);
D.total_pca = D.total_pca';
T.total_pca = (W*T.total')';
%% multiple discriminant analysis (MDA)
[D.total_mda, D.targets, w, J_W] = MultipleDiscriminantAnalysis(D.total_pca', D.targets, length(Uclasses));
D.total_mda = D.total_mda';
T.total_mda = (w'*T.total_pca')';
%% PNN
for j = 1:length(sigma_pnn)
    %% sole PNN
    [predictions.pnn(j,:)] = PNN(D.total_mda', D.targets, T.total_mda', sigma_pnn(j));
    for i = 1:length(Uclasses)
        tp_plus_fp.pnn(j,i) = sum(predictions.pnn(j,:)' == Uclasses(i));
        tp_plus_fn.pnn(j,i) = sum(T.targets == Uclasses(i));
        recall.pnn(j,i) = sum((predictions.pnn(j,:)' == Uclasses(i)) & (T.targets == Uclasses(i)))/tp_plus_fn.pnn(j,i);
        precision.pnn(j,i) = sum((predictions.pnn(j,:)' == Uclasses(i)) & (T.targets == Uclasses(i)))/tp_plus_fp.pnn(j,i);
    end
    accuracy.pnn(j) = sum(predictions.pnn(j,:)' == T.targets)/length(T.targets);
    %% PNN with voting scheme
    for i = 1:size(D.total_mda,2)
        [predictions.pnn_vote{j}(i,:)] = PNN(D.total_mda(:,i)', D.targets, T.total_mda(:,i)', sigma_pnn(j));
    end
    predictions.pnn_vote{j} = mode(predictions.pnn_vote{j});
    for i = 1:length(Uclasses)
        tp_plus_fp.pnn_vote(j,i) = sum(predictions.pnn_vote{j}' == Uclasses(i));
        tp_plus_fn.pnn_vote(j,i) = sum(T.targets == Uclasses(i));
        recall.pnn_vote(j,i) = sum((predictions.pnn_vote{j}' == Uclasses(i)) & (T.targets == Uclasses(i)))/tp_plus_fn.pnn_vote(j,i);
        precision.pnn_vote(j,i) = sum((predictions.pnn_vote{j}' == Uclasses(i)) & (T.targets == Uclasses(i)))/tp_plus_fp.pnn_vote(j,i);
    end
    accuracy.pnn_vote(j) = sum(predictions.pnn_vote{j}' == T.targets)/length(T.targets);
end
%% Knn
for j = 1:length(Knn)
    [predictions.knn(j,:)] = Nearest_Neighbor(D.total_mda', D.targets, T.total_mda', Knn(j)); %, param_struct.knn{j}
    for i = 1:length(Uclasses)
        tp_plus_fp.knn(j,i) = sum(predictions.knn(j,:)' == Uclasses(i));
        tp_plus_fn.knn(j,i) = sum(T.targets == Uclasses(i));
        recall.knn(j,i) = sum((predictions.knn(j,:)' == Uclasses(i)) & (T.targets == Uclasses(i)))/tp_plus_fn.knn(j,i);
        precision.knn(j,i) = sum((predictions.knn(j,:)' == Uclasses(i)) & (T.targets == Uclasses(i)))/tp_plus_fp.knn(j,i);
    end
    accuracy.knn(j) = sum(predictions.knn(j,:)' == T.targets)/length(T.targets);
end
%%
clearvars i j temp start stop J_W w mu UW W formatSpec fileName fileID S classes
