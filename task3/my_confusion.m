function [ CM,acc ] = my_confusion( Ctrues,Cpreds )
    N = size(Ctrues,1);
    CM = zeros(26,26);
    mysum = 0;
    for index = 1:N   
        CM(Ctrues(index),Cpreds(index)) = CM(Ctrues(index),Cpreds(index))+1;
    end
    
    for index = 1:26
        mysum = mysum + CM(index,index);
    end
    acc = mysum /sum(sum(CM,1),2);
end

