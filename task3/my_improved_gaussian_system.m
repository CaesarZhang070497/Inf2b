load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1688201/data.mat')    % load the data from designated place



Xtrn = double((dataset.train.images))/255.0;                               % store the training samples in Xtrn
[M,~] = size(Xtrn);                                                        % store the sixe of training sample in M
Ctrn = dataset.train.labels;                                               % store the label of training samples in Ctrn
Xtst = double(dataset.test.images)/255.0 ;
Ctrues = dataset.test.labels;
tic;
Cpreds = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst);                  % call the 'my_improved_gaussian_classify' function
toc;
%elapsedtime = toc;

 
[ c,acc ] = my_confusion( Ctrues,Cpreds );

    filename = sprintf('cm_improved.mat');
    N = size(Xtst,1);
    o = (1-acc)*N;                                                        %calculate the number of error
  
    Nerrs = o;
    T = table(N,Nerrs,acc);    
    save(filename,'c')
    disp(T);