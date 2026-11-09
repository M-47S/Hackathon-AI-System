package main

import (
	"fmt"
	"github.com/gin-gonic/gin"
)

func Hello(ctx *gin.Context){
	wr := ctx.Writer
	wr.Write([]byte("Hello"))
}


func main(){
	r := gin.Default()
	r.GET("api/hello", Hello)
	fmt.Print("The server was opened at http://localhost:8080")
	r.Run(":8080")
}