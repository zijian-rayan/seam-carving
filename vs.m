close all;
clear;
clc;
nbr=0;
%% test
% matrix=[0 1 0 0 0;0 1,0 0 0;0 1 0 0 0;0 1 0 0 0;0 1 0 0 0];
% imwrite(mat2gray(matrix), 'matrix.jpg');
%% code
for iteratif=1:100
    img=imread('banc - add2.jpg');

    %gray img;%
    imggray=rgb2gray(img);
    %figure(1)
    %gradient
    hy=fspecial('sobel');
    hx=hy';
    Iy=imfilter(double(imggray),hy,'replicate');
    Ix=imfilter(double(imggray),hx,'replicate');
    Gradient=abs(Ix)+abs(Iy);
%     subplot(211)
%     imagesc(Gradient)
%     colormap(bone)
%     colorbar()
%     add = sum(Gradient);
%     subplot(212)
%     plot(add)
%     colorbar()
    %% energy
    
    [m,n]=size(imggray);
    t=4*m*n-2*m-2*n+1+1;
    src=zeros(1,t);
    dst=zeros(1,t);
    weight=zeros(1,t);
    pos=1;
    for k=1:m  %first col & last col
        src(pos)=1;
        dst(pos)=(k-1)*n+3;
        weight(pos)=1000;
        pos=pos+1;

        src(pos)=k*n+2;
        dst(pos)=2;
        weight(pos)=1000;
        pos=pos+1;
    end

    for i=1:m-1  %main part
        for j=1:n-1
            nd = (i-1) * n + j + 2;%nd=nbpxl+2
            src(pos)=nd;
            src(pos+1)=nd+1;
            src(pos+2)=nd+1;
            src(pos+3)=nd+n+1;

            dst(pos)=nd+1;
            dst(pos+1)=nd;
            dst(pos+2)=nd+n;
            dst(pos+3)=nd;

            weight(pos)=Gradient(i,j);
            weight(pos+1)=1000;
            weight(pos+2)=1000;
            weight(pos+3)=1000;
            pos=pos+4;
        end
    end
    
    for h=1:n-1    %last line
        src(pos)=(m-1)*n+h+2;
        src(pos+1)=(m-1)*n+h+1+2;
        dst(pos)=(m-1)*n+h+1+2;
        dst(pos+1)=(m-1)*n+h+2;
        weight(pos)=Gradient(m,k);
        weight(pos+1)=1000;
        pos=pos+2;
    end
    
    %% flow
    G=digraph(src,dst,weight);
    [mf,N,S,T]=maxflow(G,1,2,'searchtrees');
    nbr=nbr+1;
    fprintf('nbr= %d,  mf=  %d\n',nbr,mf);
    im_display = img;
    cut_step=5;
    cut=draw_cut(imggray,S,cut_step);
    %imshow(cut);
    im1=im_display;
    im1(:,:,1)=im1(:,:,1)+uint8(cut);
    figure(2)
    imshow(im1);
    for i=1:m
        cut_lin=find(cut(i,:)==255);
        im2(i,:,:)=[img(i,1:(cut_lin(1)-1),:),img(i,(cut_lin(cut_step)+1):size(imggray,2),:)];
    end
    im_display=im2;
    clear im2;
    clear im1;
    clear cut_lin;
    imshow(im_display);
    imwrite((im_display), 'banc - add2.jpg');

end  %iterator


    %% draw cut
    
    
function cut=draw_cut(img,S,pixel_cut)
    cut=zeros([size(img,2),size(img,1)]);
    for i=1:size(S,1)
        if (S(i)>2)
            cut(S(i)-2)=255;
        end
    end
    cut=cut';
    test=sum(cut,1);
    p=find(test<255*size(img,1));
    if (p(1)>pixel_cut)
        cut_p=[cut(:,(1+pixel_cut):size(img,2)),zeros([size(img,1),pixel_cut])];
        cut=cut-cut_p;
    else
        cut_p=[255*ones([size(img,1),pixel_cut]),cut(:,1:size(img,2)-pixel_cut)];
        cut=cut_p-cut;
    end
end

    