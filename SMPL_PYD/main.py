import json
import numpy as np
import PySMPL

def loadJSON(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    # 1) Build SMPL model by reading model.json
    smplModel = PySMPL.buildSMPLModel()

    # 2) 모션 JSON 로드
    motionJson = loadJSON("walking_20240125.json")
    if not isinstance(motionJson, list) or len(motionJson) == 0:
        print("Motion JSON is empty or not an array!")
        return

    # 3) motionJson -> numpy 배열 (motionData)로 변환
    numFrame = len(motionJson)
    motionDim = len(motionJson[0])
    motionData = np.zeros((numFrame, motionDim), dtype=float)

    for i in range(numFrame):
        for j in range(motionDim):
            motionData[i, j] = motionJson[i][j]

    # 4) betas 초기화 (300차원)
    betas = np.zeros(300)

    # 5) 각 프레임 처리
    # for i in range(numFrame):
    for i in range(numFrame):
        # (0,1,2) → trans
        trans = motionData[i, 0:3]

        # (3 ~ motionDim-1) → thetas (약간 offset 1e-5 추가)
        thetas = motionData[i, 3:] + 1e-5

        # SMPL 계산
        verts, joints = PySMPL.SMPL_Calc(thetas, betas, smplModel)

        # 여기서 verts, joints에 대한 후처리(파일 저장/시각화 등) 가능
        # 예: print(verts.shape, joints.shape)

        # 첫 번째 프레임일 때만, verts 정보 출력 (frame1verts.csv 와 비교해볼것)
        if i == 0:
            print("[Frame 0] verts.shape =", verts.shape)
            print("[Frame 0] verts (first 10 rows):")
            print(verts[:10])

if __name__ == "__main__":
    main()
