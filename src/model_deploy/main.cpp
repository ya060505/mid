#include "mbed.h"

#include <cmath>

#include "DA7212.h"

#include "accelerometer_handler.h"

#include "config.h"

#include "magic_wand_model_data.h"

#include "uLCD_4DGL.h"


#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"

#define bufferLength (32)

#define signalLength (196)

DA7212 audio;

Serial pc(USBTX, USBRX);

InterruptIn sw2(SW2);

InterruptIn sw3(SW3);

DigitalOut led(LED1);
DigitalOut led2(LED2);
DigitalOut led3(LED3);

uLCD_4DGL uLCD(D1, D0, D2);

// The gesture index of the prediction

volatile int gesture_index;

volatile float signal0[signalLength];

volatile char serialInBuffer[bufferLength];

volatile int serialCount = 0;

volatile int state = 3, modesel, modenum, songsel, songnum = 1;

volatile int16_t waveform[kAudioTxBufferSize];

Thread t1(osPriorityNormal, 100 * 1024 /*120K stack size*/);
Thread t2;
Thread t3;
Thread t4;
Thread t5;
Thread t6;

EventQueue queue2(32 * EVENTS_EVENT_SIZE);
EventQueue queue3(32 * EVENTS_EVENT_SIZE);
EventQueue queue4(32 * EVENTS_EVENT_SIZE);
EventQueue queue5(32 * EVENTS_EVENT_SIZE);
EventQueue queue6(32 * EVENTS_EVENT_SIZE);


volatile int song0[42];

volatile int song1[24];

volatile int song2[32];

volatile int noteLength0[42];

volatile int noteLength1[24];

volatile int noteLength2[32];

void loadSignal(void)

{

  led3 = 0;

  int i = 0;

  serialCount = 0;

  audio.spk.pause();

  while(i < signalLength)

  {

    if(pc.readable())

    {

      serialInBuffer[serialCount] = pc.getc();

      serialCount++;

      if(serialCount == 5)

      {

        serialInBuffer[serialCount] = '\0';

        signal0[i] = (float) atof(serialInBuffer);

        serialCount = 0;

        i++;

      }

    }

  }

  led3 = 1;

  for(int i=0; i<42; i++) {
    song0[i] = (int) (signal0[i]*1000);
  }
  for(int i=0; i<42; i++) {
    noteLength0[i] = (int) (signal0[i+42]*10);
  }
  for(int i=0; i<24; i++) {
    song1[i] = (int) (signal0[i+84]*1000);
  }
  for(int i=0; i<24; i++) {
    noteLength1[i] = (int) (signal0[i+108]*10);
  }
  for(int i=0; i<32; i++) {
    song2[i] = (int) (signal0[i+132]*1000);
  }
  for(int i=0; i<32; i++) {
    noteLength2[i] = (int) (signal0[i+164]*10);
  }

}

void playNote(int freq)
{

  for(int i = 0; i < kAudioTxBufferSize; i++)

  {

    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));

  }

  audio.spk.play(waveform, kAudioTxBufferSize);

}

// Return the result of the last prediction

int PredictGesture(float* output) {

  // How many times the most recent gesture has been matched in a row

  static int continuous_count = 0;

  // The result of the last prediction

  static int last_predict = -1;


  // Find whichever output has a probability > 0.8 (they sum to 1)

  int this_predict = -1;

  for (int i = 0; i < label_num; i++) {

    if (output[i] > 0.8) this_predict = i;

  }


  // No gesture was detected above the threshold

  if (this_predict == -1) {

    continuous_count = 0;

    last_predict = label_num;

    return label_num;

  }


  if (last_predict == this_predict) {

    continuous_count += 1;

  } else {

    continuous_count = 0;

  }

  last_predict = this_predict;


  // If we haven't yet had enough consecutive matches for this gesture,

  // report a negative result

  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {

    return label_num;

  }

  // Otherwise, we've seen a positive result, so clear all our variables

  // and report it

  continuous_count = 0;

  last_predict = -1;


  return this_predict;

}

void dnn() {

  // Create an area of memory to use for input, output, and intermediate arrays.

  // The size of this will depend on the model you're using, and may need to be

  // determined by experimentation.

  constexpr int kTensorArenaSize = 60 * 1024;

  uint8_t tensor_arena[kTensorArenaSize];


  // Whether we should clear the buffer next time we fetch data

  bool should_clear_buffer = false;

  bool got_data = false;


  // Set up logging.

  static tflite::MicroErrorReporter micro_error_reporter;

  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  error_reporter->Report("Set up xxxxxxxxxx...\n");///////////////


  // Map the model into a usable data structure. This doesn't involve any

  // copying or parsing, it's a very lightweight operation.

  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {

    error_reporter->Report(

        "Model provided is schema version %d not equal "

        "to supported version %d.",

        model->version(), TFLITE_SCHEMA_VERSION);

    return;

  }


  // Pull in only the operation implementations we need.

  // This relies on a complete list of all the ops needed by this graph.

  // An easier approach is to just use the AllOpsResolver, but this will

  // incur some penalty in code space for op implementations that are not

  // needed by this graph.

  static tflite::MicroOpResolver<6> micro_op_resolver;

  micro_op_resolver.AddBuiltin(

      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,

      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,

                               tflite::ops::micro::Register_MAX_POOL_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,

                               tflite::ops::micro::Register_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,

                               tflite::ops::micro::Register_FULLY_CONNECTED());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,

                               tflite::ops::micro::Register_SOFTMAX());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,

                               tflite::ops::micro::Register_RESHAPE(), 1);

  // Build an interpreter to run the model with

  static tflite::MicroInterpreter static_interpreter(

      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);

  tflite::MicroInterpreter* interpreter = &static_interpreter;


  // Allocate memory from the tensor_arena for the model's tensors

  interpreter->AllocateTensors();


  // Obtain pointer to the model's input tensor

  TfLiteTensor* model_input = interpreter->input(0);

  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||

      (model_input->dims->data[1] != config.seq_length) ||

      (model_input->dims->data[2] != kChannelNumber) ||

      (model_input->type != kTfLiteFloat32)) {

    error_reporter->Report("Bad input tensor parameters in model");

    return;

  }


  int input_length = model_input->bytes / sizeof(float);


  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

  if (setup_status != kTfLiteOk) {

    error_reporter->Report("Set up failed\n");

    return;

  }

  error_reporter->Report("Set up successful...\n");

  while (true) {

    // Attempt to read new data from the accelerometer

    got_data = ReadAccelerometer(error_reporter, model_input->data.f,

                                 input_length, should_clear_buffer);


    // If there was no new data,

    // don't try to clear the buffer again and wait until next time

    if (!got_data) {

      should_clear_buffer = false;

      continue;

    }


    // Run inference, and report any error

    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {

      error_reporter->Report("Invoke failed on index: %d\n", begin_index);

      continue;

    }


    // Analyze the results to obtain a prediction

    gesture_index = PredictGesture(interpreter->output(0)->data.f);


    // Clear the buffer next time we read data

    should_clear_buffer = gesture_index < label_num;


    // Produce an output

    if (gesture_index < label_num) {

      error_reporter->Report(config.output_message[gesture_index]);

    }

  }
}

void isr3_1() {
  modesel = 0;
}

void isr3_2() {
  songsel = 0;
}

void isr2() {

  led = 0;
  state = 1;
  modenum = 1;
  modesel = 1;
  sw3.rise(isr3_1);

  while(modesel){
    if(gesture_index == 0 && modenum > 0){
      modenum --;
      wait(0.5);
    }
    else if(gesture_index == 0 && modenum == 0){
      modenum = 2;
      wait(0.5);
    }
    else if(gesture_index == 1 && modenum < 2){
      modenum ++;
      wait(0.5);
    }
    else if(gesture_index == 1 && modenum == 2){
      modenum = 0;
      wait(0.5);
    }
  }

  led = 1;

  if(modenum == 0){
    if(songnum > 0)
      songnum--;
    else if(songnum == 0)
      songnum = 2;
    state = 3;
    return;
  }
  else if(modenum == 2){
    if(songnum < 2)
      songnum++;
    else if(songnum == 2)
      songnum = 0;
    state = 3;
    return;
  }

  led2 = 0;
  state = 2;
  songnum = 1;
  songsel = 1;
  sw3.rise(isr3_2);
  
  while(songsel){
    if(gesture_index == 0 && songnum > 0){
      songnum --;
      wait(0.5);
    }
    else if(gesture_index == 0 && songnum == 0){
      songnum = 2;
      wait(0.5);
    }
    else if(gesture_index == 1 && songnum < 2){
      songnum ++;
      wait(0.5);
    }
    else if(gesture_index == 1 && songnum == 2){
      songnum = 0;
      wait(0.5);
    }
  }

  led2 = 1;
  state = 3;
}

void player()
{
  t4.start(callback(&queue4, &EventQueue::dispatch_forever));
  while(1) {
    for(int i = 0; state == 3 && i < 42; i++)
    {
      int length;
      if(songnum == 0)
        length = noteLength0[i];
      else if(songnum == 1)
        length = noteLength1[i];
      else if(songnum == 2)
        length = noteLength2[i];
      
      while(length--)
      {
        // the loop below will play the note for the duration of 1s
        for(int j = 0; state == 3 && j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
        {
          if(songnum == 0)
            queue4.call(playNote, song0[i]);
          else if(songnum == 1)
            queue4.call(playNote, song1[i]);
          else if(songnum == 2)
            queue4.call(playNote, song2[i]);
        }
        wait(1.0);
        audio.spk.pause();
      }
    }
    audio.spk.pause();
  }
}

void display()
{
  while(1) {
    uLCD.cls();
    uLCD.printf("\nstate=%d\n", state);
    if(state == 3)
      uLCD.printf("\nSONG=%d\n", songnum);
    if(state == 1 && modenum == 0){
      uLCD.printf("\n      *   \n");
      uLCD.printf("\n    *     \n");
      uLCD.printf("\n  *       \n");
      uLCD.printf("\n    *     \n");
      uLCD.printf("\n      *   \n");
    }
    else if(state == 1 && modenum == 1)
      uLCD.printf("\nSONG SELECTION\n");
    else if(state == 1 && modenum == 2){
      uLCD.printf("\n  *       \n");
      uLCD.printf("\n    *     \n");
      uLCD.printf("\n      *   \n");
      uLCD.printf("\n    *     \n");
      uLCD.printf("\n  *       \n");
    }

    if(state == 2 && songnum == 0){
      uLCD.printf("\nSELECT SONG 0\n");
    }
    else if(state == 2 && songnum == 1)
      uLCD.printf("\nSELECT SONG 1\n");
    else if(state == 2 && songnum == 2){
      uLCD.printf("\nSELECT SONG 2\n");
    }
    
    uLCD.printf("\nsong0[0]=%d\n", song0[0]);
    uLCD.printf("\nLength2[31]=%d\n", noteLength2[31]);
    wait(0.5);
  }
}

void loadSignalHandler(void) {queue6.call(loadSignal);}

int main() {

  led = 1;
  led2 = 1;

  t6.start(callback(&queue6, &EventQueue::dispatch_forever));

  queue6.call(loadSignalHandler);

  t1.start(dnn);

  t2.start(callback(&queue2, &EventQueue::dispatch_forever));

  t3.start(callback(&queue3, &EventQueue::dispatch_forever));

  t5.start(callback(&queue5, &EventQueue::dispatch_forever));

  sw2.rise(queue2.event(isr2));

  sw3.rise(queue3.event(player));

  queue5.call(display);


}