package main

import (
	"log"
	"time"
	"github.com/go-resty/resty/v2"
	"gorm.io/gorm"
)


var globalDBCon *gorm.DB
var globalAIClient = resty.New().SetTimeout(30 * time.Second)

func main(){

	SetupDB()
	InitTables()
	r := SetupRoutes()
	
	log.Print("The server was opened at http://localhost:8080")
	r.Run(":8080")
}
