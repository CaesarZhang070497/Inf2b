

load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1688201/data.mat')

epsilon=0.01;

Xtrn = double(dataset.train.images)/255.0;                                 % load the data from designated place
Ctrn = dataset.train.labels;
Xtst = double(dataset.test.images)/255.0 ;
Ctrues = dataset.test.labels;
tic;
[Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon);          % call my_gaussian_classify
toc;


   

 [c,acc] = my_confusion(Ctrues,Cpreds);
 N = size(Xtst,1);
 o = (1-acc)*N;                                                        %calculate the number of error
  
 Nerrs = o;
    T = table(N,Nerrs,acc);                                                 %create the table with these parameters
    filename = sprintf('cm.mat');
    filename1 = sprintf('m26.mat');
    filename2 = sprintf('cov26.mat');
    d = Ms(:,26);
    e = Covs(:,:,26);
    save(filename,'c');
    save(filename1,'d');
    save(filename2,'e');
    disp(T)