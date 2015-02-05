function [ output_args ] = gen_graph( connectivity_constant)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
num_vert = 10;
A = zeros(num_vert,num_vert);
idx1 = 1;
idx2 = 1;
while(idx1<=num_vert)
    while(idx2<=num_vert)
        if(rand<(connectivity_constant/2))
            A(idx1,idx2)=1;
            A(idx2,idx1)=1;
        end
        idx2=idx2+1;
    end
    idx1=idx1+1;
    idx2=1;
end
idx1 = 1;
coordinates = zeros(num_vert,2);
while(idx1<=num_vert)
    coordinates(idx1,1)=rand*2;
    coordinates(idx1,2)=rand*2;
    idx1=idx1+1;
end
figure();
 gplot(A,coordinates,'-*');

end

