function [ CM,acc ] = my_confusion( Ctrues,Cpreds )
    classes = size(unique(Ctrues),1); 
    N = size(Ctrues,1);
    CM = zeros(classes,classes);
    mysum = 0;
    for index = 1:N   
        CM(Ctrues(index),Cpreds(index)) = CM(Ctrues(index),Cpreds(index))+1;
    end
    
    for index = 1:classes
        mysum = mysum + CM(index,index);
    end
    acc = mysum /sum(sum(CM,1),2);
end

