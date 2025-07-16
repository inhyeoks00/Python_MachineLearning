#!/usr/bin/env python3
"""
에이프릴 감정 표현 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agent_april import AprilAgent

def test_emotion_expressions():
    """감정 표현 테스트"""
    april = AprilAgent(enable_tts=False)  # TTS 비활성화
    
    # 테스트할 감정별 텍스트
    test_cases = [
        ("기쁨 (joy)", "와! 정말 신나고 행복해! 이런 좋은 일이 있다니!"),
        ("화남 (angry)", "진짜 화나네! 이건 너무 못된 짓이잖아!"),
        ("슬픔 (sorrow)", "너무 슬프고 마음이 아파... 이런 일이 있다니..."),
        ("재미 (fun)", "하하하 완전 재밌다! 또 하고 싶은데?"),
        ("놀람 (surprised)", "헉! 이게 무슨 일이야? 완전 깜짝 놀랐어!")
    ]
    
    print("에이프릴 감정 분석 테스트를 시작합니다.")
    print("각 감정에 따라 다른 키가 입력됩니다:")
    print("joy -> 1, angry -> 2, sorrow -> 3, fun -> 4, surprised -> 5\n")
    
    for emotion_name, text in test_cases:
        print(f"\n=== {emotion_name} 테스트 ===")
        print(f"테스트 텍스트: {text}")
        
        # 감정 분석 실행
        detected_emotion = april.analyze_emotion(text)
        print(f"감지된 감정: {detected_emotion}")
        
        input("\n다음 테스트를 진행하려면 Enter를 누르세요...")

if __name__ == "__main__":
    test_emotion_expressions()
