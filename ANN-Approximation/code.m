clear;
clc;
load("data.mat")
% TEAM_0 - an example
%%
% presentation of the problem - points --> "not" a funtion (black box)
figure
plot(TEAM_0(:,1), TEAM_0(:,2),'k')
hold on
plot(TEAM_0(:,1), TEAM_0(:,2),'*r','MarkerSize',5,'LineWidth',1)
%%
% data sorting
[value, index] = sort(TEAM_0(:,1));
for i=1:1:length(TEAM_0')
    sort_TEAM_0 = TEAM_0(index,:);
end
figure
plot(sort_TEAM_0(:,1), sort_TEAM_0(:,2),'k')
hold on
plot(TEAM_0(:,1), TEAM_0(:,2),'*r','MarkerSize',5,'LineWidth',1)
%%
% chooseing the training data set
%
hm = 10; % def. the numer of points to chose
%
chosen = randperm(50);
for i=1:1:hm+2
    if i==1
        tds_TEAM_0(i,:) = sort_TEAM_0(1,:); % Training Data Set
    elseif i==hm+2
        tds_TEAM_0(i,:) = sort_TEAM_0(length(sort_TEAM_0),:);
    else
        tds_TEAM_0(i,:) = TEAM_0(chosen(i+1),:);
    end
end
figure
plot(TEAM_0(:,1), TEAM_0(:,2),'*r','MarkerSize',3,'LineWidth',1)
hold on
plot(tds_TEAM_0(:,1), tds_TEAM_0(:,2),'og','MarkerSize',5,'LineWidth',2)
%%
figure
plot(tds_TEAM_0(:,1), tds_TEAM_0(:,2))
hold on
plot(tds_TEAM_0(:,1), tds_TEAM_0(:,2),'og','MarkerSize',5,'LineWidth',2)
%%
% preparing training data
in_tds_TEAM_0 = tds_TEAM_0(:,1)';
target_tds_TEAM_0 = tds_TEAM_0(:,2)';
%%
% preparing testing data - all points
in_all_tds_TEAM_0 = sort_TEAM_0(:,1)';
target_all_tds_TEAM_0 = sort_TEAM_0(:,2)';
%%
% Radial basis neural network - exact fit
%
%%
net_rbef12_1_0_TEAM_0 = newrbe(in_tds_TEAM_0,target_tds_TEAM_0);
% checking correctness of operation
out_net_rbef12_1_0_TEAM_0 = sim(net_rbef12_1_0_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_rbef12_1_0_TEAM_0 = target_all_tds_TEAM_0 - out_net_rbef12_1_0_TEAM_0; % output error 
error_sum_net_rbef12_1_0_TEAM_0 = sum(abs(error_net_rbef12_1_0_TEAM_0));
%%
% network training (to minimize the differences between predicted and target output)
net_rbef12_1_0_TEAM_0 = train(net_rbef12_1_0_TEAM_0,in_tds_TEAM_0,target_tds_TEAM_0);
% !!!
%%
% NET STRUCTURE
how_many_inputs = net_rbef12_1_0_TEAM_0.numInputs
how_many_layers = net_rbef12_1_0_TEAM_0.numLayers
how_many_outputs = net_rbef12_1_0_TEAM_0.numOutputs
how_many_neurons_L1 = net_rbef12_1_0_TEAM_0.inputWeights{1}.size
how_many_neurons_L2 = net_rbef12_1_0_TEAM_0.layerWeights{2}.size
%%
weights_L1 = net_rbef12_1_0_TEAM_0.IW{1}
weights_L2 = net_rbef12_1_0_TEAM_0.LW{2}
biases_L1 = net_rbef12_1_0_TEAM_0.b{1}
biases_L2 = net_rbef12_1_0_TEAM_0.b{2}
%%
% Radial basis neural network - exact fit; spreed = 2
%
net_rbef12_2_0_TEAM_0 = newrbe(in_tds_TEAM_0,target_tds_TEAM_0,2);
% checking correctness of operation
out_net_rbef12_2_0_TEAM_0 = sim(net_rbef12_2_0_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_rbef12_2_0_TEAM_0 = target_all_tds_TEAM_0 - out_net_rbef12_2_0_TEAM_0; % output error 
error_sum_net_rbef12_2_0_TEAM_0 = sum(abs(error_net_rbef12_2_0_TEAM_0));
%%
% comparision with first net
d_1_0_2_0_error_net_rbef12_TEAM_0 = error_sum_net_rbef12_1_0_TEAM_0 - error_sum_net_rbef12_2_0_TEAM_0;
%%
% Radial basis neural network - exact fit; spreed = 0.5
% in our case larger spreed gives us smaller error
%
net_rbef12_0_5_TEAM_0 = newrbe(in_tds_TEAM_0,target_tds_TEAM_0,0.5);
% checking correctness of operation
out_net_rbef12_0_5_TEAM_0 = sim(net_rbef12_0_5_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_rbef12_0_5_TEAM_0 = target_all_tds_TEAM_0 - out_net_rbef12_0_5_TEAM_0; % output error 
error_sum_net_rbef12_0_5_TEAM_0 = sum(abs(error_net_rbef12_0_5_TEAM_0));
%%
% comparision with first net
d_1_0_0_5_error_net_rbef12_TEAM_0 = error_sum_net_rbef12_1_0_TEAM_0 - error_sum_net_rbef12_0_5_TEAM_0;
%%
% Radial basis neural network - exact fit; spreed = 5
%
net_rbef12_5_0_TEAM_0 = newrbe(in_tds_TEAM_0,target_tds_TEAM_0,5);
% checking correctness of operation
out_net_rbef12_5_0_TEAM_0 = sim(net_rbef12_5_0_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_rbef12_5_0_TEAM_0 = target_all_tds_TEAM_0 - out_net_rbef12_5_0_TEAM_0; % output error 
error_sum_net_rbef12_5_0_TEAM_0 = sum(abs(error_net_rbef12_5_0_TEAM_0));
%%
% comparision with first net
d_1_0_5_0_error_net_rbef12_TEAM_0 = error_sum_net_rbef12_1_0_TEAM_0 - error_sum_net_rbef12_5_0_TEAM_0;
%%
% Radial basis neural network - exact fit; spreed = 10
%
net_rbef12_10_0_TEAM_0 = newrbe(in_tds_TEAM_0,target_tds_TEAM_0,10);
% checking correctness of operation
out_net_rbef12_10_0_TEAM_0 = sim(net_rbef12_10_0_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_rbef12_10_0_TEAM_0 = target_all_tds_TEAM_0 - out_net_rbef12_10_0_TEAM_0; % output error 
error_sum_net_rbef12_10_0_TEAM_0 = sum(abs(error_net_rbef12_10_0_TEAM_0));
%%
% comparision with first net
d_1_0_10_0_error_net_rbef12_TEAM_0 = error_sum_net_rbef12_1_0_TEAM_0 - error_sum_net_rbef12_10_0_TEAM_0;
%%
figure
plot(TEAM_0(:,1), TEAM_0(:,2),'*r','MarkerSize',3,'LineWidth',1)
hold on
plot(tds_TEAM_0(:,1), tds_TEAM_0(:,2),'og','MarkerSize',5,'LineWidth',2)
plot(in_all_tds_TEAM_0, out_net_rbef12_1_0_TEAM_0,'.m','MarkerSize',4,'LineWidth',3)
plot(in_all_tds_TEAM_0, out_net_rbef12_2_0_TEAM_0,'.b','MarkerSize',4,'LineWidth',3)
plot(in_all_tds_TEAM_0, out_net_rbef12_0_5_TEAM_0,'.c','MarkerSize',4,'LineWidth',3)
plot(in_all_tds_TEAM_0, out_net_rbef12_5_0_TEAM_0,'.k','MarkerSize',4,'LineWidth',3)
plot(in_all_tds_TEAM_0, out_net_rbef12_10_0_TEAM_0,'.k','MarkerSize',4,'LineWidth',3)
legend('all points','chosen points','rb ef spreed=1.0','rb ef spreed=2.0','rb ef spreed=0.5','rb ef spreed=5.0');
%%
%%
% Radial basis neural network - fewer neurons
%
net_rbfn12_1_0_TEAM_0 = newrb(in_tds_TEAM_0,target_tds_TEAM_0);
% checking correctness of operation
out_net_rbfn12_1_0_TEAM_0 = sim(net_rbfn12_1_0_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_rbfn12_1_0_TEAM_0 = target_all_tds_TEAM_0 - out_net_rbfn12_1_0_TEAM_0; % output error 
error_sum_net_rbfn12_1_0_TEAM_0 = sum(abs(error_net_rbfn12_1_0_TEAM_0));
%%
how_many_inputs = net_rbfn12_1_0_TEAM_0.numInputs
how_many_layers = net_rbfn12_1_0_TEAM_0.numLayers
how_many_outputs = net_rbfn12_1_0_TEAM_0.numOutputs
how_many_neurons_L1 = net_rbfn12_1_0_TEAM_0.inputWeights{1}.size
how_many_neurons_L2 = net_rbfn12_1_0_TEAM_0.layerWeights{2}.size
%%
weights_L1 = net_rbfn12_1_0_TEAM_0.IW{1}
weights_L2 = net_rbfn12_1_0_TEAM_0.LW{2}
biasys_L1 = net_rbfn12_1_0_TEAM_0.b{1}
biasys_L2 = net_rbfn12_1_0_TEAM_0.b{2}
%%
% comparision with first net
d_error_net_rbef12_1_0_rbfn12_1_0_TEAM_0 = error_sum_net_rbef12_1_0_TEAM_0 - error_sum_net_rbfn12_1_0_TEAM_0;
%%
% Radial basis neural network - fewer neurons; spreed = 5
net_rbfn12_5_0_TEAM_0 = newrb(in_tds_TEAM_0,target_tds_TEAM_0,[0],5);
% checking correctness of operation
out_net_rbfn12_5_0_TEAM_0 = sim(net_rbfn12_5_0_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_rbfn12_5_0_TEAM_0 = target_all_tds_TEAM_0 - out_net_rbfn12_5_0_TEAM_0; % output error 
error_sum_net_rbfn12_5_0_TEAM_0 = sum(abs(error_net_rbfn12_5_0_TEAM_0));
%%
how_many_inputs = net_rbfn12_5_0_TEAM_0.numInputs
how_many_layers = net_rbfn12_5_0_TEAM_0.numLayers
how_many_outputs = net_rbfn12_5_0_TEAM_0.numOutputs
how_many_neurons_L1 = net_rbfn12_5_0_TEAM_0.inputWeights{1}.size
how_many_neurons_L2 = net_rbfn12_5_0_TEAM_0.layerWeights{2}.size
%%
weights_L1 = net_rbfn12_5_0_TEAM_0.IW{1}
weights_L2 = net_rbfn12_5_0_TEAM_0.LW{2}
biasys_L1 = net_rbfn12_5_0_TEAM_0.b{1}
biasys_L2 = net_rbfn12_5_0_TEAM_0.b{2}
%%
% comparision with first net
d_error_net_rbef12_5_0_rbfn12_5_0_TEAM_0 = error_sum_net_rbef12_5_0_TEAM_0 - error_sum_net_rbfn12_5_0_TEAM_0;
%%
%%
% Radial basis neural network - fewer neurons; spreed = 5; goal = 3!!!
net_rbfn12_3_0_5_0_TEAM_0 = newrb(in_tds_TEAM_0,target_tds_TEAM_0,[3.0],5);
% checking correctness of operation
out_net_rbfn12_3_0_5_0_TEAM_0 = sim(net_rbfn12_3_0_5_0_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_rbfn12_3_0_5_0_TEAM_0 = target_all_tds_TEAM_0 - out_net_rbfn12_3_0_5_0_TEAM_0; % output error 
error_sum_net_rbfn12_3_0_5_0_TEAM_0 = sum(abs(error_net_rbfn12_3_0_5_0_TEAM_0));
%%
how_many_inputs = net_rbfn12_3_0_5_0_TEAM_0.numInputs
how_many_layers = net_rbfn12_3_0_5_0_TEAM_0.numLayers
how_many_outputs = net_rbfn12_3_0_5_0_TEAM_0.numOutputs
how_many_neurons_L1 = net_rbfn12_3_0_5_0_TEAM_0.inputWeights{1}.size
how_many_neurons_L2 = net_rbfn12_3_0_5_0_TEAM_0.layerWeights{2}.size
%%
weights_L1 = net_rbfn12_3_0_5_0_TEAM_0.IW{1}
weights_L2 = net_rbfn12_3_0_5_0_TEAM_0.LW{2}
biasys_L1 = net_rbfn12_3_0_5_0_TEAM_0.b{1}
biasys_L2 = net_rbfn12_3_0_5_0_TEAM_0.b{2}
%%
% comparision with first net
d_error_net_rbef12_5_0_rbfn12_3_0_5_0_TEAM_0 = error_sum_net_rbef12_5_0_TEAM_0 - error_sum_net_rbfn12_3_0_5_0_TEAM_0;
%%
% Radial basis neural network - fewer neurons; spreed = 1; goal = 3!!!
net_rbfn12_3_0_1_0_TEAM_0 = newrb(in_tds_TEAM_0,target_tds_TEAM_0,[3.0],1);
% checking correctness of operation
out_net_rbfn12_3_0_1_0_TEAM_0 = sim(net_rbfn12_3_0_1_0_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_rbfn12_3_0_1_0_TEAM_0 = target_all_tds_TEAM_0 - out_net_rbfn12_3_0_1_0_TEAM_0; % output error 
error_sum_net_rbfn12_3_0_1_0_TEAM_0 = sum(abs(error_net_rbfn12_3_0_1_0_TEAM_0));
%%
how_many_inputs = net_rbfn12_3_0_1_0_TEAM_0.numInputs
how_many_layers = net_rbfn12_3_0_1_0_TEAM_0.numLayers
how_many_outputs = net_rbfn12_3_0_1_0_TEAM_0.numOutputs
how_many_neurons_L1 = net_rbfn12_3_0_1_0_TEAM_0.inputWeights{1}.size
how_many_neurons_L2 = net_rbfn12_3_0_1_0_TEAM_0.layerWeights{2}.size
%%
weights_L1 = net_rbfn12_3_0_1_0_TEAM_0.IW{1}
weights_L2 = net_rbfn12_3_0_1_0_TEAM_0.LW{2}
biasys_L1 = net_rbfn12_3_0_1_0_TEAM_0.b{1}
biasys_L2 = net_rbfn12_3_0_1_0_TEAM_0.b{2}
%%
% comparision with first net
d_error_net_rbef12_5_0_rbfn12_3_0_1_0_TEAM_0 = error_sum_net_rbef12_5_0_TEAM_0 - error_sum_net_rbfn12_3_0_1_0_TEAM_0;
d_error_net_rbef12_1_0_rbfn12_3_0_1_0_TEAM_0 = error_sum_net_rbef12_1_0_TEAM_0 - error_sum_net_rbfn12_3_0_1_0_TEAM_0;

%%
% Forward BackPropagation
%
%%
%%
% FeedForward 12(tansing) 1(pureline)
net_ffbp12_TEAM_0 = newff(in_tds_TEAM_0,target_tds_TEAM_0,[12]);
% checking correctness of operation
out_net_ffbp12_TEAM_0 = sim(net_ffbp12_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_ffbp12_TEAM_0 = target_all_tds_TEAM_0 - out_net_ffbp12_TEAM_0; % output error 
error_sum_net_ffbp12_TEAM_0 = sum(abs(error_net_ffbp12_TEAM_0));
%%
% network training
net_ffbp12_TEAM_0 = train(net_ffbp12_TEAM_0,in_tds_TEAM_0,target_tds_TEAM_0);
out_net_ffbp12_TEAM_0 = sim(net_ffbp12_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_ffbp12_TEAM_0 = target_all_tds_TEAM_0 - out_net_ffbp12_TEAM_0; % output error 
error_sum_net_ffbp12_TEAM_0 = sum(abs(error_net_ffbp12_TEAM_0));
%%
% NET STRUCTURE
how_many_inputs = net_ffbp12_TEAM_0.numInputs
how_many_layers = net_ffbp12_TEAM_0.numLayers
how_many_outputs = net_ffbp12_TEAM_0.numOutputs
how_many_neurons_L1 = net_ffbp12_TEAM_0.inputWeights{1}.size
how_many_neurons_L2 = net_ffbp12_TEAM_0.layerWeights{2}.size
%%
weights_L1 = net_ffbp12_TEAM_0.IW{1}
weights_L2 = net_ffbp12_TEAM_0.LW{2}
biasys_L1 = net_ffbp12_TEAM_0.b{1}
biasys_L2 = net_ffbp12_TEAM_0.b{2}
%%
% comparision with first net
d_error_net_rbef12_5_0_ffbp12_TEAM_0 = error_sum_net_rbef12_5_0_TEAM_0 - error_sum_net_ffbp12_TEAM_0;
%%
figure
plot(TEAM_0(:,1), TEAM_0(:,2),'*r','MarkerSize',3,'LineWidth',1)
hold on
plot(tds_TEAM_0(:,1), tds_TEAM_0(:,2),'og','MarkerSize',5,'LineWidth',2)
plot(in_all_tds_TEAM_0, out_net_rbef12_1_0_TEAM_0,'.m','MarkerSize',4,'LineWidth',3)
plot(in_all_tds_TEAM_0, out_net_ffbp12_TEAM_0,'.m','MarkerSize',4,'LineWidth',3)
legend('all points','chosen points','rb ef spreed=1.0','ffbp 12');

%%
% FeedForward 8(tansing) 4(tansing) 1(pureline)
clc
net_ffbp8_4_TEAM_0 = newff(in_tds_TEAM_0,target_tds_TEAM_0,[8 4]);
% checking correctness of operation
out_net_ffbp8_4_TEAM_0 = sim(net_ffbp8_4_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_ffbp8_4_TEAM_0 = target_all_tds_TEAM_0 - out_net_ffbp8_4_TEAM_0; % output error 
error_sum_net_ffbp8_4_TEAM_0 = sum(abs(error_net_ffbp8_4_TEAM_0));
%%
% network training
net_ffbp8_4_TEAM_0 = train(net_ffbp8_4_TEAM_0,in_tds_TEAM_0,target_tds_TEAM_0);
out_net_ffbp8_4_TEAM_0 = sim(net_ffbp8_4_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_ffbp8_4_TEAM_0 = target_all_tds_TEAM_0 - out_net_ffbp8_4_TEAM_0; % output error 
error_sum_net_ffbp8_4_TEAM_0 = sum(abs(error_net_ffbp8_4_TEAM_0));
%%
figure
plot(TEAM_0(:,1), TEAM_0(:,2),'*r','MarkerSize',3,'LineWidth',1)
hold on
plot(tds_TEAM_0(:,1), tds_TEAM_0(:,2),'og','MarkerSize',5,'LineWidth',2)
plot(in_all_tds_TEAM_0, out_net_ffbp12_TEAM_0,'.m','MarkerSize',4,'LineWidth',3)
plot(in_all_tds_TEAM_0, out_net_ffbp8_4_TEAM_0,'.c','MarkerSize',4,'LineWidth',3)
legend('all points','chosen points','ffbp 12(tansig) 1(pureline)','ffbp 8(tansig) 4(tansig) 1(pureline)');
%%
% FeedForward 6(tansing) 4(tansing) 2(tansing) 1(pureline)
clc
net_ffbp6_4_2_TEAM_0 = newff(in_tds_TEAM_0,target_tds_TEAM_0,[6 4 2]);
% checking correctness of operation
out_net_ffbp6_4_2_TEAM_0 = sim(net_ffbp6_4_2_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_ffbp6_4_2_TEAM_0 = target_all_tds_TEAM_0 - out_net_ffbp6_4_2_TEAM_0; % output error 
error_sum_net_ffbp6_4_2_TEAM_0 = sum(abs(error_net_ffbp6_4_2_TEAM_0));
%%
% network training
net_ffbp6_4_2_TEAM_0 = train(net_ffbp6_4_2_TEAM_0,in_tds_TEAM_0,target_tds_TEAM_0);
out_net_ffbp6_4_2_TEAM_0 = sim(net_ffbp6_4_2_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_ffbp6_4_2_TEAM_0 = target_all_tds_TEAM_0 - out_net_ffbp6_4_2_TEAM_0; % output error 
error_sum_net_ffbp6_4_2_TEAM_0 = sum(abs(error_net_ffbp6_4_2_TEAM_0));
%%
figure
plot(TEAM_0(:,1), TEAM_0(:,2),'*r','MarkerSize',3,'LineWidth',1)
hold on
plot(tds_TEAM_0(:,1), tds_TEAM_0(:,2),'og','MarkerSize',5,'LineWidth',2)
plot(in_all_tds_TEAM_0, out_net_ffbp12_TEAM_0,'.m','MarkerSize',4,'LineWidth',3)
plot(in_all_tds_TEAM_0, out_net_ffbp8_4_TEAM_0,'.c','MarkerSize',4,'LineWidth',3)
plot(in_all_tds_TEAM_0, out_net_ffbp6_4_2_TEAM_0,'.b','MarkerSize',4,'LineWidth',3)

legend('all points','chosen points','ffbp 12(tansig) 1(pureline)','ffbp 8(tansig) 4(tansig) 1(pureline)','ffbp 6(tansig) 4(tansig) 2(tansig) 1(pureline)');
%%
%
%%
% CascadeForward 12(tansing) 1(pureline)
net_cfbp12_TEAM_0 = newcf(in_tds_TEAM_0,target_tds_TEAM_0,[12]);
% checking correctness of operation
out_net_cfbp12_TEAM_0 = sim(net_cfbp12_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_cfbp12_TEAM_0 = target_all_tds_TEAM_0 - out_net_cfbp12_TEAM_0; % output error 
error_sum_net_cfbp12_TEAM_0 = sum(abs(error_net_cfbp12_TEAM_0));
%%
% network training
net_cfbp12_TEAM_0 = train(net_cfbp12_TEAM_0,in_tds_TEAM_0,target_tds_TEAM_0);
out_net_cfbp12_TEAM_0 = sim(net_cfbp12_TEAM_0,in_all_tds_TEAM_0); % simulation
error_net_cfbp12_TEAM_0 = target_all_tds_TEAM_0 - out_net_cfbp12_TEAM_0; % output error 
error_sum_net_cfbp12_TEAM_0 = sum(abs(error_net_cfbp12_TEAM_0));
%%
% NET STRUCTURE
how_many_inputs = net_cfbp12_TEAM_0.numInputs
how_many_layers = net_cfbp12_TEAM_0.numLayers
how_many_outputs = net_cfbp12_TEAM_0.numOutputs
how_many_neurons_L1 = net_cfbp12_TEAM_0.inputWeights{1}.size
how_many_neurons_L2 = net_cfbp12_TEAM_0.layerWeights{2}.size
%%
weights_L1 = net_cfbp12_TEAM_0.IW{1}
weights_L2 = net_cfbp12_TEAM_0.LW{2}
biasys_L1 = net_cfbp12_TEAM_0.b{1}
biasys_L2 = net_ffbp12_TEAM_0.b{2}
%%
% comparision with first net
d_error_net_rbef12_5_0_cfbp12_TEAM_0 = error_sum_net_rbef12_5_0_TEAM_0 - error_sum_net_cfbp12_TEAM_0;
%%
figure
plot(TEAM_0(:,1), TEAM_0(:,2),'*r','MarkerSize',3,'LineWidth',1)
hold on
plot(tds_TEAM_0(:,1), tds_TEAM_0(:,2),'og','MarkerSize',5,'LineWidth',2)
plot(in_all_tds_TEAM_0, out_net_rbef12_1_0_TEAM_0,'.m','MarkerSize',4,'LineWidth',3)
plot(in_all_tds_TEAM_0, out_net_ffbp12_TEAM_0,'.c','MarkerSize',4,'LineWidth',3)
plot(in_all_tds_TEAM_0, out_net_cfbp12_TEAM_0,'.m','MarkerSize',4,'LineWidth',3)
legend('all points','chosen points','rb ef spreed=1.0','ffbp 12','cfbp 12');


%In neural networks, transfer functions are used to introduce nonlinearity into the model. The transfer function takes in the weighted sum of inputs to the neuron and produces an output. Two commonly used transfer functions are the tanh (hyperbolic tangent) and purelin (linear) functions.
%The tanh function maps the input to a value between -1 and 1, and has a sigmoidal shape with an S-curve. It is commonly used as an activation function in hidden layers of neural networks.
%The purelin function is simply the identity function, which maps the input to the output without any transformation. It is typically used as the activation function in the output layer of neural networks when the output should be a linear combination of the inputs.
