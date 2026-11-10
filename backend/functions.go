package main

import (
	"log"
	"os"
	"strconv"
	"time"
	"github.com/joho/godotenv"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

func InitTables(){
	err := globalDBCon.AutoMigrate()
	if err != nil {
		log.Fatal("Critical error: Can't create tables")
	}
	log.Print("Successfuly init tables")
}

func SetupDB() {
	if err := godotenv.Load(); err != nil {
		log.Fatal("Critical error: Can't load ENV")
	}
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		log.Fatal("Critical error: Can't find string to connect DB")
	}
	var err error
	globalDBCon, err = gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Fatalf("Can't connect to db: %v", err)
	}
	poolDB, err := globalDBCon.DB()
	if err != nil {
		log.Fatalf("Can't get object db: %v", err)
	}

	maxOpen, _ := strconv.Atoi(os.Getenv("DB_MAX_OPEN_CONNS"))
	if maxOpen == 0 {
		maxOpen = 30
	}
	poolDB.SetMaxOpenConns(maxOpen)

	maxIdle, _ := strconv.Atoi(os.Getenv("DB_MAX_IDLE_CONNS"))
	if maxIdle == 0 {
		maxIdle = 10
	}
	poolDB.SetMaxIdleConns(maxIdle)

	poolDB.SetConnMaxLifetime(30 * time.Minute)

	poolDB.SetConnMaxIdleTime(5 * time.Minute)
	log.Print("DB succesfully connected and setted")
}

func GetAIAnwser(json_data UserRequest) (AIResponseForm, error){
	var ai_request RequestToAI
	ai_request.session_id = json_data.session_id
	ai_request.request_id = json_data.request_id
	ai_request.message = json_data.message
	//ai_request.history = db.where(...)
	//ai.request.customer_profile = ...
	var ai_response AIResponseForm
	_, err := globalAIClient.R().
				SetBody(ai_request).
				SetResult(&ai_response).
				Post("http://localhost/v1/chat")
	if err != nil {
		return AIResponseForm{}, err
	}
	return ai_response, nil
}


