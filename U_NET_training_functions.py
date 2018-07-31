import logging
import numpy as np
import os
import random
import time
from PIL import ImageTk, Image, ImageOps
import tkinter as tk

log = logging.getLogger("")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

def color_outlines_to_bw(img):
    i = Image.open(img)
    pixdata = i.load()
    for y in range(i.size[1]):
        for x in range(i.size[0]):
            [r, g, b] = pixdata[x, y]
            if (max(r, g, b) - min(r, g,
                                   b)) < 80:  ## trial and error = best fit
                pixdata[x, y] = (0, 0, 0)
            else:
                pixdata[x, y] = (255, 255, 255)
    return i

def emb_image_batch_generator_v2(data_folder, emb_list, batch_size, rsz, MIN_SIZE=64):
    MAX_SIZE = rsz[0]
    while True:
        log.info("Spawning a new embryo image batch...")
        image_section_batch = []
        pixel_labels_batch = []

        test_size = []

        while len(image_section_batch) < batch_size:
            emb_choice = random.choice(emb_list)


            flip = random.choice([True, False])

            emb_raw_image = Image.open(emb_choice["raw_image"]).convert("L").crop(emb_choice["box_coords"])
            if flip:
                emb_raw_image = ImageOps.flip(emb_raw_image)
            rotations = random.randint(0, 360)
            emb_raw_image = emb_raw_image.rotate(rotations, expand=True, resample=Image.BICUBIC)

            REQUIRED_SIZE = random.randint(MIN_SIZE, MAX_SIZE)
            if REQUIRED_SIZE > emb_raw_image.size[0] or REQUIRED_SIZE > emb_raw_image.size[1]:
                continue

            emb_outlines_image = Image.open(emb_choice["bw_outline"]).crop(emb_choice["box_coords"])
            if flip:
                emb_outlines_image = ImageOps.flip(emb_outlines_image)
            emb_outlines_image = emb_outlines_image.rotate(rotations, expand=True, resample=Image.BICUBIC)


            binary_outlines = emb_outlines_image.convert("1")
            binary_outlines_arr = np.asarray(binary_outlines).copy()


            X_LIMIT = emb_raw_image.size[0] - REQUIRED_SIZE
            Y_LIMIT = emb_raw_image.size[1] - REQUIRED_SIZE

            fail_counter = 0
            while fail_counter < 25:
                try:
                    X_RANDOM = (random.randint(0, X_LIMIT))
                    Y_RANDOM = (random.randint(0, Y_LIMIT))
                    b_box = [X_RANDOM, Y_RANDOM, X_RANDOM + REQUIRED_SIZE -1, Y_RANDOM + REQUIRED_SIZE -1]

                    w_range = range(b_box[0], b_box[2])
                    h_range = range(b_box[1], b_box[3])


                    bb_w1_pixels = [binary_outlines_arr[(b_box[1]), i] for i in
                                    w_range]
                    bb_w2_pixels = [binary_outlines_arr[(b_box[3]), i] for i in
                                    w_range]
                    bb_h1_pixels = [binary_outlines_arr[(i, b_box[0])] for i in
                                    h_range]
                    bb_h2_pixels = [binary_outlines_arr[(i, b_box[2])] for i in
                                    h_range]
                except Exception as e:
                    pass

                th = 3 # proportion to divide the edges into to check crossing
                x = [
                    any(bb_w1_pixels[:len(bb_w1_pixels) // th]) and any(
                        bb_w1_pixels[-len(bb_w1_pixels) // th:]),
                    any(bb_w2_pixels[:len(bb_w2_pixels) // th]) and any(
                        bb_w2_pixels[-len(bb_w2_pixels) // th:]),
                    any(bb_h1_pixels[:len(bb_h1_pixels) // th]) and any(
                        bb_h1_pixels[-len(bb_h1_pixels) // th:]),
                    any(bb_h2_pixels[:len(bb_h2_pixels) // th]) and any(
                        bb_h2_pixels[-len(bb_h2_pixels) // th:])
                ]

                if all(x) == True:
                    crop_emb_raw_image = emb_raw_image.crop(box=b_box)
                    crop_emb_raw_image_arr = np.asarray(
                        crop_emb_raw_image.resize((rsz[0], rsz[1]), Image.ANTIALIAS)).reshape(
                        rsz[0], rsz[1], 1)

                    image_section_batch.append(crop_emb_raw_image_arr / float(255))

                    crop_emb_outlines_image = emb_outlines_image.convert("L").crop(box=b_box)
                    crop_emb_outlines_image_arr = np.asarray(
                        crop_emb_outlines_image.resize((rsz[0], rsz[1]), Image.ANTIALIAS)).reshape(
                        rsz[0], rsz[1], 1)

                    truefalse = crop_emb_outlines_image_arr > 155

                    pixel_labels_batch.append(truefalse.astype(np.uint8))
                fail_counter += 1

            #log.info("    couldnt find crossing in {} attempts:".format(fail_counter))
            #log.info("    for image: {}".format(fail_counter))
        yield np.asarray(image_section_batch), np.asarray(pixel_labels_batch)


def get_base_dataset(data_folder, regenerate_bw=False):
    _Z_STACK = {
        "AAntnew33-47.lsm (cropped)": 3,
        "ANTERIOR \"EMB 4\" Nov. 28th Emb (2)_L5_Sum.lsm (spliced)": 3,
        "Anterior = Embryo 5\" Feb 20th": 4,
        "USEAnt(potential)2_march__t2.lsm (spliced) (cropped)": 3,
        "LATERAL \"EMB 3\" Oct 2nd Emb (1)_L3_Sum.lsm (spliced)": 3,
        "LATERAL\"EMB 6\" Nov. 28th Emb (2)_L7_Sum.lsm ": 3,
        "LATERAL \"EMB 9\" Dec 15th Emb (1)_L12_Sum.lsm (spliced)": 4,
        "LATERAL \"EMB 12\" Nov. 28th Emb (2)_L12_Sum.lsm ": 4,
        "Outline this movie tp 6-22 posterior copy": 5,
        "POSTERIOR = \"Embryo 2\", Nov. 28th Emb (2)_L3_Sum.lsm (spliced)": 3,
        "EARLY Posterior = \"Embryo 6\", Feb. 20th Emb (1)_L6_Sum.lsm (spliced)": 3,
        "LATE Posterior = \"Embryo 6\", Feb. 20th Emb (1)_L6_Sum.lsm (spliced)": 4
    }
    emb_list = [x for x in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, x))]
    raw_images = []
    raw_outlines = []
    bw_outlines = []
    validation_count = {}

    bounding_boxes = []

    for emb in emb_list:
        for t_folder in os.listdir(os.path.join(data_folder, emb)):
            if t_folder.startswith('T'):
                opt_z = _Z_STACK[emb]
                image_needed = t_folder + "C02" + "Z00" + str(opt_z) + ".tif"
                raw_images.append(os.path.join(data_folder, emb, t_folder, image_needed))

    for f in raw_images:
        filedir = os.path.join(os.path.sep.join(f.split('/')[:-1]))
        outline = os.path.join(filedir, [x for x in os.listdir(filedir) if x.startswith('xCell')][0])
        if not os.path.isfile(outline):
            # could not find associated outline file
            raise Exception()
        raw_outlines.append(outline)
        # save: this is slow, only do if unhappy with the bw outline generation method
        bw_path = os.path.join(os.path.sep.join(outline.split(os.path.sep)[:-1]), 'bw_outline.tif')
        bw_outlines.append(bw_path)

        if not os.path.exists(bw_path) or regenerate_bw:
            print("WARNING: generating bw image")
            bw_outline = color_outlines_to_bw(outline)
            if os.path.exists(bw_path):
                pass
            bw_outline.save(bw_path)

    for test_im_idx, test_im in enumerate(bw_outlines):
        scanme = np.asarray(Image.open(test_im).convert('L')).copy()
        seg = np.sum(scanme, axis=1).tolist()

        chunk_threshold = 20 # horiz or vert size of cluster of pixels below which we discard

        is_blank = True
        measure_top = 0
        measure_bottom = 0
        chunks_vert = []

        # TODO: refactor this bit to general horizontal/vertical segmentation
        # Vertical splitting
        for i, tot in enumerate(seg):
            if is_blank:
                measure_top = i
                measure_bottom = i
            else:
                measure_bottom += 1
            if tot > 0:
                is_blank = False
            else:
                is_blank = True
            if is_blank:
                if (measure_bottom - chunk_threshold) > measure_top:
                    chunks_vert.append((measure_top, measure_bottom))

        if len(chunks_vert) <= 2:
            # horizontal splitting if we dont have noise
            for cv in chunks_vert:
                _cv = scanme[cv[0]:cv[1], :]
                seg = np.sum(_cv, axis=0).tolist()
                chunk_threshold = 20
                is_blank = True
                measure_top = 0
                measure_bottom = 0
                chunks_horiz = []
                for i, tot in enumerate(seg):
                    if is_blank:
                        measure_top = i
                        measure_bottom = i
                    else:
                        measure_bottom += 1
                    if tot > 0:
                        is_blank = False
                    else:
                        is_blank = True
                    if is_blank:
                        if (measure_bottom - chunk_threshold) > measure_top:
                            chunks_horiz.append((measure_top, measure_bottom))
                if len(chunks_horiz) <= 2:
                    for ch in chunks_horiz:
                        coords = (ch[0], cv[0], ch[1], cv[1])
                        if not (test_im in validation_count):
                            bounding_boxes.append(
                                {
                                "raw_image": raw_images[test_im_idx],
                                "raw_outline": raw_outlines[test_im_idx],
                                "bw_outline": bw_outlines[test_im_idx],
                                "box_coords": coords
                                }
                            )
                            # do a unique check, see how many of original dataset we are using
                            #validation_count[test_im] = True
                        _ch = _cv[:, ch[0]:ch[1]]
                else:
                    print ("stop")
                    pass

    print("dataset images valid: {}".format(len(bounding_boxes)))
    return bounding_boxes


if __name__ == '__main__':
    # test/show the generator in action
    batch_size = 2
    min_size = 80
    size_to_resize_to = (256, 256)
    data_folder = "/home/len/dev/thesis/epithelial_cell_border_identification"
    force_regenerate_bw_images = False  # set this to true to regenrate the saved outlines if the creation method changes

    dataset = get_base_dataset(data_folder, regenerate_bw=force_regenerate_bw_images)
    training_generator = emb_image_batch_generator_v2(
            data_folder,
            dataset,
            batch_size,
            size_to_resize_to,
            MIN_SIZE=min_size)

    window = tk.Tk()
    window.title("data")
    window.geometry("1024x1024")
    window.configure(background='grey')
    canvas = tk.Canvas(window, width=2000, height=2000)
    while True:
        res = next(training_generator)
        for i, r in enumerate(res[0]):
            a = Image.fromarray((r*255).reshape(size_to_resize_to[0], size_to_resize_to[1]))
            b = Image.fromarray((res[1][i] * 255).reshape(size_to_resize_to[0], size_to_resize_to[1]))

            img = ImageTk.PhotoImage(a)
            imgb = ImageTk.PhotoImage(b)

            canvas.delete("all")
            canvas.pack()
            canvas.create_image((512, 512), image=img, anchor="center")
            canvas.create_image((512 + size_to_resize_to[0] + 5, 512), image=imgb, anchor="center")
            window.update_idletasks()
            window.update()
            time.sleep(0.1)


