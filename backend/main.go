package main

import (
	"log"
	"time"
	"github.com/go-resty/resty/v2"
)



var globalAIClient = resty.New().SetTimeout(30 * time.Second)

func main(){
	
	r := SetupRoutes()
	log.Print("The server was opened at http://localhost:8080")
	r.Run(":8080")
}

{
  "session_id": "session-123",
  "status": "active",
  "created_at": "2026-11-09T12:30:00Z"
}

{
  "session_id": "session-123",
  "request_id": "request-789",
  "message": "Показания собираем вручную",
  "history": [
    {
      "role": "user",
      "content": "У нас сеть из 50 магазинов, хотим снизить расходы"
    },
    {
      "role": "assistant",
      "content": "Как сейчас организован сбор показаний электроэнергии?"
    }
  ],
  "customer_profile": {
    "business_type": "retail",
    "objects_count": 50,
    "problem": "Высокие расходы на электроэнергию",
    "metering_method": null
  },
  "conversation_summary": null
}



{
  "session_id": "session-123",
  "request_id": "request-456",
  "message": {
    "id": "msg-002",
    "role": "assistant",
    "content": "Как сейчас организован сбор показаний электроэнергии в ваших магазинах?",
    "created_at": "2026-11-09T12:31:00Z"
  },
  "action": "ask_question",
  "products": []
}

{
  "action": "recommend",
  "message": {
    "id": "msg-002",
    "role": "assistant",
    "content": "Как сейчас организован сбор показаний электроэнергии в ваших магазинах?",
    "created_at": "2026-11-09T12:31:00Z"
  },
  "product_ids": ["cloud_askue"],
  "customer_profile": {
    "business_type": "retail",
    "objects_count": 50,
    "problem": "Высокие расходы на электроэнергию",
    "metering_method": "manual"
  },
  "conversation_summary": "Клиент управляет сетью из 50 магазинов, хочет снизить расходы на электроэнергию. Показания собираются вручную. Предварительно рекомендована облачная АСКУЭ, требуется проверить совместимость оборудования."
}