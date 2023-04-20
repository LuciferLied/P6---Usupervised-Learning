def imgsave(pic_list):
        fig = plt.figure(figsize=(42, 25))
        for i in range(4):
            plt.subplot(2, 4, i*2+1)
            plt.imshow(pic_list[0][i].cpu().squeeze().numpy().transpose(1, 2, 0))
        for i in range(4):
            plt.subplot(2, 4, i*2+2)
            plt.imshow(pic_list[1][i].cpu().squeeze().detach().numpy().transpose(1, 2, 0))
        plt.savefig('pics/epoch.png')


pic_list = [save_pic, save_dec]
imgsave(pic_list)