import { type NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import path from "path"

export async function POST(request: NextRequest) {
  try {
    const { audioData, filename } = await request.json()

    if (!audioData) {
      return NextResponse.json({ error: "No audio data provided" }, { status: 400 })
    }

    // Call MediaPipe classification script
    const scriptPath = path.join(process.cwd(), "scripts", "mediapipe_audio_classifier.py")

    return new Promise((resolve) => {

      // Use 'python' on Windows if 'python3' is not standard, but sticking to existing pattern 'python3' 
      // or better, try to detect or just use 'python' which is more standard on Windows?
      // Let's use 'python' as it is more likely correct for Windows, 
      // but if the user had 'python3' working before, maybe they have an alias.
      // I will trust the user's environment BUT passing data via stdin is the key change.

      const pythonCommand = process.platform === "win32" ? "python" : "python3"
      const python = spawn(pythonCommand, [scriptPath])

      let stdout = ""
      let stderr = ""

      python.stdout.on("data", (data) => {
        stdout += data.toString()
      })

      python.stderr.on("data", (data) => {
        stderr += data.toString()
      })

      python.on("error", (err) => {
        console.error("Failed to start python process:", err)
        resolve(NextResponse.json({ error: "Failed to start python process", details: err.message }, { status: 500 }))
      })

      python.on("close", (code) => {
        if (code === 0) {
          try {
            // The python script now prints ONLY the JSON to stdout
            const jsonResult = JSON.parse(stdout.trim())
            resolve(NextResponse.json(jsonResult))
          } catch (error) {
            console.error("JSON Parse Error:", error)
            resolve(
              NextResponse.json(
                {
                  error: "Failed to parse classification results",
                  details: error instanceof Error ? error.message : "Unknown error",
                  stdout,
                  stderr,
                },
                { status: 500 },
              ),
            )
          }
        } else {
          resolve(
            NextResponse.json(
              {
                error: "MediaPipe classification failed",
                code,
                stderr,
                stdout,
              },
              { status: 500 },
            ),
          )
        }
      })

      // Send data via STDIN
      const inputData = JSON.stringify({
        audioData,
        filename: filename || "uploaded_audio"
      })

      try {
        python.stdin.write(inputData)
        python.stdin.end()
      } catch (err) {
        console.error("Error writing to python stdin:", err)
      }

      // Set timeout for long-running processes
      setTimeout(() => {
        python.kill()
        resolve(
          NextResponse.json(
            {
              error: "Classification timeout",
              message: "MediaPipe classification took too long",
            },
            { status: 408 },
          ),
        )
      }, 60000) // 60 second timeout
    })
  } catch (error) {
    console.error("Audio classification API error:", error)
    return NextResponse.json(
      {
        error: "Internal server error",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
