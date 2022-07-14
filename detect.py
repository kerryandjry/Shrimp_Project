import argparse
import cv2
import torch
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, check_img_size, increment_path


def run(weights: str, img_path: str, project: str, conf_thres=0.25, iou_thres=0.45, agnostic_nms=False):
    save_dir = increment_path(Path(project), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    names = 'Shrimp'
    move_points = set()
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.parameters())))
    imgsz = check_img_size(640, s=stride)
    dataset = LoadImages(img_path, img_size=imgsz, stride=stride, auto=True)
    img_arr = []
    shrimp_num = 0
    shrimp_num_current_frame: int
    shrimp_num_last_frame = 0

    for path, img, img0, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device) / 255.
        if len(img.shape) == 3:
            img = img[None]
        pred = model(img, augment=False, visualize=False)[0]
        print(pred)
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic_nms)
        print("===================")
        shrimp_num_current_frame = len(*pred)
        if shrimp_num_last_frame < shrimp_num_current_frame:
            shrimp_num = shrimp_num + (shrimp_num_current_frame - shrimp_num_last_frame)
            shrimp_num_last_frame = shrimp_num_current_frame
        elif shrimp_num_last_frame > shrimp_num_current_frame:
            shrimp_num_last_frame = shrimp_num_current_frame

        for i, det in enumerate(pred):
            aims = list()
            s += f'{i}: '
            p, frame = path, getattr(dataset, 'frame', 0)
            p = Path(p)
            save_path = str(save_dir / p.name)
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                num_of_shrimp = f'TOTAL = {str(shrimp_num)}'
                cv2.putText(img0, num_of_shrimp, (500, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)
                    aim = ('%g ' * len(line)).rstrip() % line
                    aim = aim.split(' ')
                    aims.append(aim)
                    # c = int(cls)  # integer class
                    p1, p2, mid = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (
                        int(xyxy[0]) // 2 + int(xyxy[2]) // 2, int(xyxy[1]) // 2 + int(xyxy[3]) // 2)
                    move_points.add(mid)
                    cv2.putText(img0, names, p1, cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(img0, p1, p2, (255, 0, 0), thickness=max(round(sum(img0.shape) / 2 * 0.003), 2),
                                  lineType=cv2.LINE_AA)
            if vid_cap:
                for point in move_points:
                    cv2.circle(img0, point, radius=1, color=(0, 0, 255), thickness=3)

            if save_path:
                # if isinstance(vid_writer[i], cv2.VideoWriter):
                #     vid_writer[i].release()
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    img_arr.append(img0)

            cv2.imshow('result', img0)
            cv2.waitKey(1)
    # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    # for img in img_arr:
    #     vid_writer.write(img)
    # print(f'save in {save_path}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/new/best.pt')
    parser.add_argument('--img_path', type=str, default=r'/home/lee/Work/data/shrimp_video_new/2022-03-12-10_13_25.mp4')

    parser.add_argument('--project', default='runs/detect')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
