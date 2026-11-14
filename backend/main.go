package main

import (
	"log"
	"time"
	"github.com/go-resty/resty/v2"
	"gorm.io/gorm"
)


var globalDBCon *gorm.DB
var globalAIClient = resty.New().SetTimeout(30 * time.Second)

// @title           AI Chat & Leads API
// @version         1.0
// @description     API сервиса чата с ИИ и управления заявками
// @host            localhost:8080
// @BasePath        /

// @securityDefinitions.apikey CookieAuth
// @in cookie
// @name cookie_id
func main(){
	InitDB()
	SetupDBConn()
	InitTables()
	initLogger()
	r := SetupRoutes()
	
	log.Print("The server was opened at http://localhost:8080")
	r.Run(":8080")
}
