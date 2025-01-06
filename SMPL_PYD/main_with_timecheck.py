import json
import time
import numpy as np
import PySMPL

def loadJSON(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    # --------------------------------------------------
    # 1) Build SMPL model by reading model.json
    # --------------------------------------------------
    t0 = time.time()
    smplModel = PySMPL.buildSMPLModel()
    t1 = time.time()
    print(f"[TIME] buildSMPLModel() took {t1 - t0:.6f} sec")

    # --------------------------------------------------
    # 2) 모션 JSON 로드
    # --------------------------------------------------
    t0 = time.time()
    motionJson = loadJSON("walking_20240125.json")
    t1 = time.time()
    print(f"[TIME] loadJSON('walking_20240125.json') took {t1 - t0:.6f} sec")

    if not isinstance(motionJson, list) or len(motionJson) == 0:
        print("Motion JSON is empty or not an array!")
        return

    # --------------------------------------------------
    # 3) motionJson -> numpy 배열 (motionData)로 변환
    # --------------------------------------------------
    t0 = time.time()
    numFrame = len(motionJson)
    motionDim = len(motionJson[0])
    motionData = np.zeros((numFrame, motionDim), dtype=float)

    for i in range(numFrame):
        for j in range(motionDim):
            motionData[i, j] = motionJson[i][j]
    t1 = time.time()
    print(f"[TIME] motionJson -> motionData took {t1 - t0:.6f} sec")

    # --------------------------------------------------
    # 4) betas 초기화 (300차원)
    # --------------------------------------------------
    betas = np.zeros(300)

    # --------------------------------------------------
    # 5) 각 프레임 처리
    # --------------------------------------------------
    t_loop_start = time.time()
    for i in range(numFrame):
        # (0,1,2) → trans
        trans = motionData[i, 0:3]

        # (3 ~ motionDim-1) → thetas (약간 offset 1e-5 추가)
        thetas = motionData[i, 3:] + 1e-5

        # SMPL 계산 (프레임별 시간 측정)
        t2 = time.time()
        verts, joints = PySMPL.SMPL_Calc(thetas, betas, smplModel)
        t3 = time.time()

        # 주석 해제하면 각 프레임별 처리 시간도 확인 가능
        # print(f"[TIME] Frame {i} SMPL_Calc took {t3 - t2:.6f} sec")

        # 후처리(예: print(verts.shape, joints.shape)) 등...

    t_loop_end = time.time()
    print(f"[TIME] Total for all frames: {t_loop_end - t_loop_start:.6f} sec")

if __name__ == "__main__":
    main()
