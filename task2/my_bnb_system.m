load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1688201/data.mat')



Xtrn = double((dataset.train.images));
Ctrn = dataset.train.labels;
Xtst = double(dataset.test.images) ;
Ctrues = dataset.test.labels;

%threshold = 1
for threshold = 1                                                     % set the threshole to  one
    tic;
    Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold);                 % call the bnb classify function
    toc;

    N = length(Ctrues);
    [cm,acc] = my_confusion(Ctrues,Cpreds);                                  % call my confusion matrix
    o = (1-acc)*size(Xtst,1);                                                        %calculate the number of error
  
    Nerrs = o;
    T = table(N,Nerrs,acc) ;                                                %create the table with these parameters
    filename = sprintf('cm.mat');
    
    save(filename,'cm')
    disp(T);
end
