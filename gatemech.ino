#include <Servo.h>
Servo Serv;
String myCmd;
int echoPin = 9; //IR sensor digital pin
int pinServo=6; //servo motor pin
int trigPin=10;
int redPin = 3;
int greenPin = 5;
int duration;
int distance;
String one = "1";
String zero ="0";

void setup()
{
   Serv.attach(pinServo);
   pinMode(trigPin, OUTPUT); // Sets the trigPin as an OUTPUT
   pinMode(echoPin, INPUT);
   pinMode(redPin,OUTPUT);
   pinMode(greenPin,OUTPUT);
   Serial.begin(115200);
}

void loop()
{
    
   digitalWrite(trigPin, LOW);
   delayMicroseconds(2);
   // Sets the trigPin HIGH (ACTIVE) for 10 microseconds
   digitalWrite(trigPin, HIGH);
   delayMicroseconds(10);
   digitalWrite(trigPin, LOW);
   duration = pulseIn(echoPin, HIGH);
   // Calculating the distance
   distance = (duration * 0.034 / 2);
   Serial.print("Distance: ");
   Serial.print(distance);
   Serial.print("cm");
   Serial.println();
   while(Serial.available() == 0)
   {
      
   }
   myCmd = Serial.readStringUntil('\0');
   myCmd = myCmd.charAt(0);
   Serial.println(myCmd);
   if(myCmd == one)
   {
    if (distance < 12 && distance > 0) 
    { 
      Serv.write(2);
      digitalWrite(greenPin,LOW);
      digitalWrite(redPin,HIGH);
      delay(1000);
    }
    else
    {
      Serv.write(90);
      digitalWrite(redPin,LOW);
      digitalWrite(greenPin,HIGH);
      delay(8);
    }
    
   }
 }
