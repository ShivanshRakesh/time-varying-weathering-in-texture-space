import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.transform import resize


# normalize_img normalizes our output to be between 0 and 1
def normalize_img(im):
    img = im.copy()
    img += np.abs(np.min(img))
    img /= np.max(img)
    return img

def diff(block1, block2):
    details1, gradient1 = block1
    details2, gradient2 = block2
    
    dg = np.abs(gradient1 - gradient2)
    dd = np.abs(details1 - details2)
    
    if details2.ndim == 3:        
        dg = np.nan_to_num(dg/np.max(dg, axis=(-1, -2))[:, None, None])   
        dd = np.nan_to_num(dd/np.max(dd, axis=(-1, -2))[:, None, None])
    else:
        dg = np.nan_to_num(dg/np.max(dg))  
        dd = np.nan_to_num(dd/np.max(dd))

    return dg+dd

def l2_top_bottom(patch_top, patch_bottom, alpha, all_cm_blocks_target, target_cm, block_size, overlap_size):
    
    patch_top_d, patch_top_g = patch_top
    patch_bottom_d, patch_bottom_g = patch_bottom
    
    block_top_d, block_top_g = patch_top_d[-overlap_size:, :], patch_top_g[-overlap_size:, :]
    maxy = min(block_top_d.shape[1], block_size)
    
    if patch_bottom_d.ndim == 2:
        block_bottom_d, block_bottom_g = patch_bottom_d[:overlap_size], patch_bottom_g[:overlap_size]
        block_bottom_d, block_bottom_g = block_bottom_d[:, :maxy], block_bottom_g[:, :maxy]
    elif patch_bottom_d.ndim == 3:
        block_bottom_d, block_bottom_g = patch_bottom_d[:, :overlap_size], patch_bottom_g[:, :overlap_size]
        block_bottom_d, block_bottom_g = block_bottom_d[:, :, :maxy], block_bottom_g[:, :, :maxy]
    else:
        raise ValueError('patch_bottom must have 3 or 4 dimensions')

    top_cost = alpha * l2_loss([block_top_d, block_top_g], [block_bottom_d, block_bottom_g])
    y2 = min(target_cm.shape[0], block_size)
    top_cost += (1 - alpha) * corr_loss(target_cm[:y2, :], all_cm_blocks_target[:, :y2, :])

    return top_cost


def l2_left_right(patch_left, patch_right, alpha, all_cm_blocks_target, target_cm, block_size, overlap_size):
    
    patch_left_d, patch_left_g = patch_left
    patch_right_d, patch_right_g = patch_right
    
    block_left_d, block_left_g = patch_left_d[:, -overlap_size:], patch_left_g[:, -overlap_size:]

    if patch_right_d.ndim == 2:
        block_right_d, block_right_g = patch_right_d[:, :overlap_size], patch_right_g[:, :overlap_size]
    elif patch_right_d.ndim == 3:
        block_right_d, block_right_g = patch_right_d[:, :, :overlap_size], patch_right_g[:, :, :overlap_size]
    else:
        raise ValueError('patch_right must have 3 or 4 dimensions')

    # overlap error
    left_cost = alpha * l2_loss([block_left_d, block_left_g], [block_right_d, block_right_g])
    # add correspondence error
    x2 = min(target_cm.shape[1], block_size)
    left_cost += (1 - alpha) * corr_loss(target_cm[:, :x2], all_cm_blocks_target[:, :, :x2])

    return left_cost


def corr_loss(block_1, block_2):
    return np.sum(np.sum((block_1 - block_2) ** 2, axis=-1), axis=-1) ## Block1 comes from current age_map???


def l2_loss(block_1, block_2):
    sqdfs = (diff(block_1, block_2)) ** 2, ## instead of difference, value using distance function
    return np.sqrt(np.sum(np.sum(sqdfs, axis=-1), axis=-1))


def select_min_patch(patches, cost, tolerance=0.1):
    min_cost = np.min(cost)
    upper_cost_bound = min_cost + tolerance * min_cost
    # pick random patch within tolerance
    possible_vals = [np.argwhere(cost <= upper_cost_bound).flatten()]
    r = np.random.randint(len(possible_vals))
    return patches[0][r], patches[1][r], patches[2][r]


def compute_error_surface(block_1, block_2):
    return (diff(block_1, block_2)) ** 2 ## Instead of difference, value from distance function?


def min_vert_path(error_surf_vert, block_size):
    top_min_path = np.zeros(block_size, dtype=np.int)
    top_min_path[0] = np.argmin(error_surf_vert[0, :], axis=0)
    for i in range(1, block_size):
        err_mid_v = error_surf_vert[i, :]
        mid_v = err_mid_v[top_min_path[i - 1]]

        err_left = np.roll(error_surf_vert[i, :], 1, axis=0)
        err_left[0] = np.inf
        left = err_left[top_min_path[i - 1]]

        err_right = np.roll(error_surf_vert[i, :], -1, axis=0)
        err_right[-1] = np.inf
        right = err_right[top_min_path[i - 1]]

        next_poss_pts_v = np.vstack((left, mid_v, right))
        new_pts_ind_v = top_min_path[i - 1] + (np.argmin(next_poss_pts_v, axis=0) - 1)
        top_min_path[i] = new_pts_ind_v

    return top_min_path


def min_hor_path(error_surf_hor, block_size):
    left_min_path = np.zeros(block_size, dtype=np.int)
    left_min_path[0] = np.argmin(error_surf_hor[:, 0], axis=0)
    for i in range(1, block_size):
        err_mid_h = error_surf_hor[:, i]
        mid_h = err_mid_h[left_min_path[i - 1]]

        err_top = np.roll(error_surf_hor[:, i], 1, axis=0)
        err_top[0] = np.inf
        top = err_top[left_min_path[i - 1]]

        err_bot = np.roll(error_surf_hor[:, i], -1, axis=0)
        err_bot[-1] = np.inf
        bot = err_bot[left_min_path[i - 1]]

        next_poss_pts_h = np.vstack((top, mid_h, bot))
        new_pts_ind_h = left_min_path[i - 1] + (np.argmin(next_poss_pts_h, axis=0) - 1)
        left_min_path[i] = new_pts_ind_h

    return left_min_path


def compute_lr_join(block_left, block_right, block_size, overlap_size, error_surf_vert=None):
    if error_surf_vert is None:
        error_surf_vert = compute_error_surface([block_right[1], block_right[2]], [block_left[1], block_left[2]])

    vert_path = min_vert_path(error_surf_vert, block_size)
    yy, xx = np.meshgrid(np.arange(block_size), np.arange(overlap_size))
    vert_mask = xx.T <= np.tile(np.expand_dims(vert_path, 1), overlap_size)

    lr_join = np.zeros_like(block_left[0])
    lr_join[:, :][vert_mask] = block_left[0][vert_mask]
    lr_join[:, :][~vert_mask] = block_right[0][~vert_mask]

    lr_join_d = np.zeros_like(block_left[1])
    lr_join_d[:, :][vert_mask] = block_left[1][vert_mask]
    lr_join_d[:, :][~vert_mask] = block_right[1][~vert_mask]

    lr_join_g = np.zeros_like(block_left[2])
    lr_join_g[:, :][vert_mask] = block_left[2][vert_mask]
    lr_join_g[:, :][~vert_mask] = block_right[2][~vert_mask]

    return lr_join, lr_join_d, lr_join_g


def compute_bt_join(block_top, block_bottom, block_size, overlap_size, error_surf_hor=None):
    if error_surf_hor is None:
        error_surf_hor = compute_error_surface([block_bottom[1], block_bottom[2]], [block_top[1], block_top[2]])

    hor_path = min_hor_path(error_surf_hor, block_size)
    yy, xx = np.meshgrid(np.arange(block_size), np.arange(overlap_size))
    hor_mask = (xx.T <= np.tile(np.expand_dims(hor_path, 1), overlap_size)).T

    bt_join = np.zeros_like(block_top[0])
    bt_join[:, :][hor_mask] = block_top[0][hor_mask]
    bt_join[:, :][~hor_mask] = block_bottom[0][~hor_mask]

    bt_join_d = np.zeros_like(block_top[1])
    bt_join_d[:, :][hor_mask] = block_top[1][hor_mask]
    bt_join_d[:, :][~hor_mask] = block_bottom[1][~hor_mask]

    bt_join_g = np.zeros_like(block_top[2])
    bt_join_g[:, :][hor_mask] = block_top[2][hor_mask]
    bt_join_g[:, :][~hor_mask] = block_bottom[2][~hor_mask]

    return bt_join, bt_join_d, bt_join_g


def lr_bt_join_double(best_left_block, right_block, best_top_block, bottom_block, block_size, overlap_size):
    error_surf_hor = compute_error_surface([best_left_block[1], best_left_block[2]], [right_block[1], right_block[2]])

    maxy = min(bottom_block[0].shape[1], block_size)
    best_top_block = [best_top_block[0][:, :maxy], best_top_block[1][:, :maxy], best_top_block[2][:, :maxy]]
    # best_top_block[1] = best_top_block[1][:, :maxy]
    # best_top_block[2] = best_top_block[2][:, :maxy]
    error_surf_vert = compute_error_surface([best_top_block[1], best_top_block[2]], [bottom_block[1], bottom_block[2]])

    vert_contrib = np.zeros_like(error_surf_vert)
    hor_contrib = np.zeros_like(error_surf_hor)

    vert_contrib[:, :overlap_size] += (error_surf_hor[:overlap_size, :] + error_surf_vert[:, :overlap_size]) / 2
    hor_contrib[:overlap_size, :] += (error_surf_vert[:, :overlap_size] + error_surf_hor[:overlap_size, :]) / 2

    error_surf_vert += vert_contrib
    error_surf_hor += hor_contrib

    left_right_join, left_right_join_d, left_right_join_g = compute_lr_join(right_block, best_left_block, block_size, overlap_size, error_surf_vert=error_surf_hor)
    bottom_top_join, bottom_top_join_d, bottom_top_join_g = compute_bt_join(bottom_block, best_top_block, block_size, overlap_size, error_surf_hor=error_surf_vert)

    return left_right_join, left_right_join_d, left_right_join_g, bottom_top_join, bottom_top_join_d, bottom_top_join_g


def transfer_texture(texture_src, img_target, C, block_size):

    overlap_size = int(block_size / 6)
    texture_src, d_src, g_src = texture_src
    img_target, d_target, g_target = img_target
    src_cmap, target_cmap = C

    h, w, c = texture_src.shape

    assert block_size < min(h, w)

    dh, dw = h * 2, w * 2

    nx_blocks = ny_blocks = max(dh, dw) // block_size
    w_new = h_new = nx_blocks * block_size - (nx_blocks - 1) * overlap_size

    img_target = resize(img_target, (h_new, w_new), preserve_range=True)
    target = img_target.copy()
    target_cmap = resize(target_cmap, (h_new, w_new), preserve_range=True)
    target_d = resize(d_target, (h_new, w_new), preserve_range=True)
    target_g = resize(g_target, (h_new, w_new), preserve_range=True)

    n = 5
    for i in range(n):

        osz = int(block_size / 6)

        assert block_size < min(h, w)

        y_max, x_max = h - block_size, w - block_size

        xs = np.arange(x_max)
        ys = np.arange(y_max)
        all_blocks = np.array([texture_src[y:y + block_size, x:x + block_size] for x in xs for y in ys])
        all_blocks_d = np.array([d_src[y:y + block_size, x:x + block_size] for x in xs for y in ys])
        all_blocks_g = np.array([g_src[y:y + block_size, x:x + block_size] for x in xs for y in ys])
        # all_cm_blocks_target = np.sum(all_blocks, axis=-1)  ## Source age map??
        all_cm_blocks_target = np.array([src_cmap[y:y + block_size, x:x + block_size] for x in xs for y in ys])

        # img_target = resize(img_target, (h_new, w_new), preserve_range=True)
        y_begin = 0
        y_end = block_size

        alpha_i = 0.8 * (i / (n - 1)) + 0.1

        print('alpha = %.2f, block size = %d' % (alpha_i, block_size))
        step = block_size - osz

        for y in range(ny_blocks):

            x_begin = 0
            x_end = block_size

            for x in range(nx_blocks):
                if x == 0 and y == 0:
                    # randomly select top left patch
                    r = np.random.randint(len(all_blocks))
                    random_patch, random_patch_d, random_patch_g = all_blocks[r], all_blocks_d[r], all_blocks_g[r]
                    target[y_begin:y_end, x_begin:x_end], target_d[y_begin:y_end, x_begin:x_end], target_g[y_begin:y_end, x_begin:x_end] = random_patch, random_patch_d, random_patch_g

                    x_begin = x_end
                    x_end += step

                    continue

                xa, xb = x_begin - block_size, x_begin
                ya, yb = y_begin - block_size, y_begin

                if y == 0:
                    y1 = 0
                    y2 = block_size
                    left_patch, left_patch_d, left_patch_g = target[y1:y2, xa:xb], target_d[y1:y2, xa:xb], target_g[y1:y2, xa:xb]
                    left_block, left_block_d, left_block_g = left_patch[:, -osz:], left_patch_d[:, -osz:], left_patch_g[:, -osz:]

                    left_patch_cm = target_cmap[y2 - block_size:y2, x_end - block_size:x_end]

                    left_cost = l2_left_right(patch_left=[left_patch_d, left_patch_g], patch_right=[all_blocks_d, all_blocks_g],
                                              alpha=alpha_i, target_cm = left_patch_cm,
                                              all_cm_blocks_target=all_cm_blocks_target, 
                                              block_size=block_size, overlap_size=overlap_size)

                    best_right_patch, best_right_patch_d, best_right_patch_g = select_min_patch([all_blocks, all_blocks_d, all_blocks_g], left_cost)
                    best_right_block, best_right_block_d, best_right_block_g = best_right_patch[:, :osz], best_right_patch_d[:, :osz], best_right_patch_g[:, :osz]

                    # join left and right blocks
                    left_right_join, left_right_join_d, left_right_join_g = compute_lr_join(
                                                                        [left_block, left_block_d, left_block_g], 
                                                                        [best_right_block, best_right_block_d, best_right_block_g], 
                                                                        block_size, overlap_size)
                    
                    full_join = np.hstack(
                        (target[y1:y2, xa:xb - osz], left_right_join, best_right_patch[:, osz:]))
                    full_join_d = np.hstack(
                        (target_d[y1:y2, xa:xb - osz], left_right_join_d, best_right_patch_d[:, osz:]))
                    full_join_g = np.hstack(
                        (target_g[y1:y2, xa:xb - osz], left_right_join_g, best_right_patch_g[:, osz:]))

                    xm = target[y1:y2, xa:x_end].shape[1]
                    target[y1:y2, xa:x_end], target_d[y1:y2, xa:x_end], target_g[y1:y2, xa:x_end] = full_join[:, :xm], full_join_d[:, :xm], full_join_g[:, :xm]
                else:
                    if x == 0:
                        x1 = 0
                        x2 = block_size

                        top_patch, top_patch_d, top_patch_g = target[ya:yb, x1:x2], target_d[ya:yb, x1:x2], target_g[ya:yb, x1:x2]
                        top_block, top_block_d, top_block_g = top_patch[-osz:, :], top_patch_d[-osz:, :], top_patch_g[-osz:, :]

                        top_patch_cm = target_cmap[y_end - block_size:y_end, x2 - block_size:x2]

                        top_cost = l2_top_bottom(patch_top=[top_patch_d, top_patch_g], patch_bottom=[all_blocks_d, all_blocks_g],
                                                 alpha=alpha_i, target_cm = top_patch_cm,
                                                 all_cm_blocks_target=all_cm_blocks_target, 
                                                 block_size=block_size, overlap_size=overlap_size)
                        best_bottom_patch, best_bottom_patch_d, best_bottom_patch_g = select_min_patch([all_blocks, all_blocks_d, all_blocks_g], top_cost)
                        best_bottom_block, best_bottom_block_d, best_bottom_block_g = best_bottom_patch[:osz, :], best_bottom_patch_d[:osz, :], best_bottom_patch_g[:osz, :]

                        # join top and bottom blocks
                        top_bottom_join, top_bottom_join_d, top_bottom_join_g = compute_bt_join(
                                                                        [top_block, top_block_d, top_block_g], 
                                                                        [best_bottom_block, best_bottom_block_d, best_bottom_block_g], 
                                                                        block_size, overlap_size)
                        
                        full_join = np.vstack(
                            (target[ya:yb - osz, x1:x2], top_bottom_join, best_bottom_patch[osz:, :]))                        
                        full_join_d = np.vstack(
                            (target_d[ya:yb - osz, x1:x2], top_bottom_join_d, best_bottom_patch_d[osz:, :]))                        
                        full_join_g = np.vstack(
                            (target_g[ya:yb - osz, x1:x2], top_bottom_join_g, best_bottom_patch_g[osz:, :]))

                        xm = target[ya:y_end, x1:x2].shape[1]
                        target[ya:y_end, x1:x2], target_d[ya:y_end, x1:x2], target_g[ya:y_end, x1:x2] = full_join[:, :xm], full_join_d[:, :xm], full_join_g[:, :xm]
                    else:
                        # overlap is L-shaped
                        y1, y2 = y_begin - osz, y_end
                        x1, x2 = x_begin - osz, x_end

                        left_patch, left_patch_d, left_patch_g = target[y1:y2, xa:xb], target_d[y1:y2, xa:xb], target_g[y1:y2, xa:xb]
                        left_block, left_block_d, left_block_g = left_patch[:, -osz:], left_patch_d[:, -osz:], left_patch_g[:, -osz:]

                        top_patch, top_patch_d, top_patch_g = target[ya:yb, x1:x2], target_d[ya:yb, x1:x2], target_g[ya:yb, x1:x2]
                        top_block, top_block_d, top_block_g = top_patch[-osz:, :], top_patch_d[-osz:, :], top_patch_g[-osz:, :]

                        curr_patch_cm = target_cmap[y2 - block_size:y2, x_end - block_size:x_end]

                        left_cost = l2_left_right(patch_left=[left_patch_d, left_patch_g], patch_right=[all_blocks_d, all_blocks_g],
                                                  alpha=alpha_i, target_cm = curr_patch_cm,
                                                  all_cm_blocks_target=all_cm_blocks_target, 
                                                  block_size=block_size, overlap_size=overlap_size)

                        top_cost = l2_top_bottom(patch_top=[top_patch_d, top_patch_g], patch_bottom=[all_blocks_d, all_blocks_g],
                                                 alpha=alpha_i, target_cm = curr_patch_cm,
                                                 all_cm_blocks_target=all_cm_blocks_target, 
                                                 block_size=block_size, overlap_size=overlap_size)

                        best_right_patch, best_right_patch_d, best_right_patch_g = best_bottom_patch, best_bottom_patch_d, best_bottom_patch_g = select_min_patch([all_blocks, all_blocks_d, all_blocks_g], top_cost + left_cost)

                        best_right_block, best_right_block_d, best_right_block_g = best_right_patch[:, :osz], best_right_patch_d[:, :osz], best_right_patch_g[:, :osz]
                        best_bottom_block, best_bottom_block_d, best_bottom_block_g = best_bottom_patch[:osz, :], best_bottom_patch_d[:osz, :], best_bottom_patch_g[:osz, :]

                        left_right_join, left_right_join_d, left_right_join_g, top_bottom_join, top_bottom_join_d, top_bottom_join_g = lr_bt_join_double(
                                                                            [best_right_block, best_right_block_d, best_right_block_g], [left_block, left_block_d, left_block_g],
                                                                            [best_bottom_block, best_bottom_block_d, best_bottom_block_g], [top_block, top_block_d, top_block_g], 
                                                                            block_size, overlap_size)
                        # join left and right blocks
                        full_lr_join = np.hstack(
                            (target[y1:y2, xa:xb - osz], left_right_join, best_right_patch[:, osz:]))
                        full_lr_join_d = np.hstack(
                            (target_d[y1:y2, xa:xb - osz], left_right_join_d, best_right_patch_d[:, osz:]))
                        full_lr_join_g = np.hstack(
                            (target_g[y1:y2, xa:xb - osz], left_right_join_g, best_right_patch_g[:, osz:]))

                        # join top and bottom blocks
                        full_tb_join = np.vstack(
                            (target[ya:yb - osz, x1:x2], top_bottom_join, best_bottom_patch[osz:, :]))                        
                        full_tb_join_d = np.vstack(
                            (target_d[ya:yb - osz, x1:x2], top_bottom_join_d, best_bottom_patch_d[osz:, :]))                        
                        full_tb_join_g = np.vstack(
                            (target_g[ya:yb - osz, x1:x2], top_bottom_join_g, best_bottom_patch_g[osz:, :]))

                        target[ya:y_end, x1:x2], target_d[ya:y_end, x1:x2], target_g[ya:y_end, x1:x2] = full_tb_join, full_tb_join_d, full_tb_join_g
                        target[y1:y2, xa:x_end], target_d[y1:y2, xa:x_end], target_g[y1:y2, xa:x_end] = full_lr_join, full_lr_join_d, full_lr_join_g

                x_begin = x_end
                x_end += step
                if x_end > w_new:
                    x_end = w_new

            y_begin = y_end
            y_end += step

            if y_end > h_new:
                y_end = h_new
    return target


def show_text_trans(img):
    plt.title('Texture Transfer')
    plt.imshow(normalize_img(img))
    plt.axis('off')
    plt.show()


# source_texture = plt.imread('data/texture14.jpg').astype(np.float32)
# target_image = plt.imread('data/bill.jpg').astype(np.float32)

# block_size = 30
# overlap_size = int(block_size / 6)

# show_text_trans(transfer_texture(source_texture, target_image, block_size))
