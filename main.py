import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

app = FastAPI()

# API 키설정
API_KEY = "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"

# 로컬 Ollama 클라이언트 연결
local_client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 클라우드 추론 API 클라이언트 연결
cloud_client = openai.OpenAI(api_key=API_KEY)

#  1차 백엔드에서 넘겨줄 데이터 임시 정의
class RawFocusData(BaseModel):
    study_time: int # 총 공부 시간
    average_focus_score: int # 평균 집중도 점수
    gaze_distraction_count: int # 시선이탈 횟수
    bad_posture_count: int  # 불량 자세 횟수
    smartphone_count: int # 스마트폰 감지 횟수

@app.post("/analyze-hybrid")
async def analyze_focus_hybrid(data: RawFocusData):
    try:
        # 로컬 일반 모델(Gemma 3)  방대한 Raw 로그 데이터 정제/요약
        print(" Gemma 3를 이용한 Raw 데이터 정제 및 요약 중...") 
        local_prompt = (
            "아래의 학생 실시간 집중도 로그 배열을 분석해서, "
            "사용자의 총 공부시간에 따른 집중도 분석을 각 데이터의 숫자를 기반하고 포함해서 2문장 이상으로 요약해줘.\n"
            f"총 공부시간 : {data.study_time}분 , 평균 집중도 점수 {data.average_focus_score}점 ,시선이탈 횟수 {data.gaze_distraction_count}회, 불량자세 횟수 {data.bad_posture_count}, 스마트폰 감지 횟수 {data.smartphone_count}  "
        )
        
        local_response = local_client.chat.completions.create(
            model="gemma4:e4b",
            messages=[
                {"role": "system", "content": "너는 방대한 로그 데이터를 핵심만 압축하는 데이터 정제 전문가야."},
                {"role": "user", "content": local_prompt}
            ]
        )
        
        local_summary = local_response.choices[0].message.content
        print(f"[Gemma 3 정제 결과]: {local_summary}\n")

  
        # 정제된 요약본을 들고 외부 고성능 추론 API(o3-mini 등) 호출하기

        print("정제된 데이터를 바탕으로 외부 고성능 추론 API 호출 중...")
        
        cloud_system_instruction = (
            "너는 인공지능 기반 학습 집중도 분석 전문가야. 사용자의 정량적인 학습 데이터(집중도 점수, 시선 이탈, 자세 흐트러짐, 스마트폰 사용 등)를 분석하여 깊이 있는 맞춤형 피드백을 제공하는 것이 너의 임무야. "
            "분석은 냉철하고 객관적이어야 하며, 해결책(솔루션)은 따뜻하고 실천 가능해야 해."
            "제공된 정보를 바탕으로 사용자의 [강점, 약점, 원인분석, 맞춤 피드백]을 도출해줘. "
            "반드시 다른 설명 없이 오직 아래의 JSON 포맷으로만 응답해줘:\n"
            '{"strength": "...", "weakness": "...", "cause": "...", "feedback": "..."}'
        )
        
        cloud_user_content = (
            f"학생 기본 데이터: 공부시간 {data.study_time}분, 평균 집중도 점수 {data.average_focus_score}점\n"
            f"로컬 AI가 분석한 집중도 시계열 요약: {local_summary}"
        )
        
        cloud_response = cloud_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},  # JSON 강제
            messages=[
                {"role": "system", "content": cloud_system_instruction},
                {"role": "user", "content": cloud_user_content}
            ]
        )
        
        final_result = json.loads(cloud_response.choices[0].message.content)
        print("[최종 피드백 생성 완료]")
        
        # 프론트엔드 대시보드와 DB로 넘겨줄 최종 결과 반환
        return {
            "status": "success",
            "intermediate_clean_data": local_summary, 
            "final_analysis": final_result # 추론 모델이 만든 최종 피드백
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파이프라인 가동 실패: {str(e)}")
