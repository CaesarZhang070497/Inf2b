
load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1688201/data.mat')    %load the dataset fro, designated palce



Xtrn = single(double(dataset.train.images))/255.0;                         % convert the type to single instead of double to save memory
Ctrn = dataset.train.labels;                                               % devided by 255 as required in the handout
Xtst = single(dataset.test.images) /255.0;
ks = [1 3 5 10 20]';                                                       % the array Ks is created as instruction from handout 
Ctrues = dataset.test.labels; 
tic;
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, ks);                            % the function 'my_knn_classify' is called here.
toc;
elapsedtime = toc;

for k = 1:size(ks)                                                         % for each k fetch the corresponding column from Cpreds 
    Cpred = Cpreds(:,k);                                                   % and store the the cloumn in 'Cpred'
    [ CM, d] = my_confusion( Ctrues,Cpred );                               % call the my_confusion function with these data
    filename = sprintf('cm%d.mat',ks(k));                                      % dynamicallly create file's name
    cm = CM;
    save(filename,'cm')                                                    % save the matrix into designated file name.

    N = length(Ctrues);                                                        % total amount of classes
    o = (1-d)*size(Xtst,1);                                                            % the number of errors
    acc = d;                                                                   % the rate of accuracy
    Nerrs = o;
    n = ks(k);
    T = table(n,N,Nerrs,acc);
    disp(T);
end

