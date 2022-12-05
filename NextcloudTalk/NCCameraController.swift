//
// Copyright (c) 2022 Marcel Müller <marcel-mueller@gmx.de>
//
// Author Marcel Müller <marcel-mueller@gmx.de>
//
// GNU GPL version 3 or any later version
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

// Based on https://developer.apple.com/documentation/vision/applying_matte_effects_to_people_in_images_and_video

import Foundation
import Vision
import CoreImage.CIFilterBuiltins

@objcMembers
class NCCameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, MTKViewDelegate {

    // State
    private var backgroundBlurEnabled = true
    private var supportsBackgroundBlur = false
    private var usingFrontCamera = true // TODO: Implement camera switching
    private var deviceOrientation: UIDeviceOrientation = UIDevice.current.orientation
    private var videoRotation: RTCVideoRotation = ._0

    // AVFoundation
    private var session: AVCaptureSession?

    // WebRTC
    private var videoSource: RTCVideoSource
    private var videoCapturer: RTCVideoCapturer

    // Vision
    private let requestHandler = VNSequenceRequestHandler()
    private var facePoseRequest: VNDetectFaceRectanglesRequest!
    // Needs to be Any? as VNGeneratePersonSegmentationRequest is only supported on iOS 15 onwards
    private var segmentationRequest: Any?

    // Metal
    private var metalDevice: MTLDevice!
    private var metalCommandQueue: MTLCommandQueue!

    public weak var localView: MTKView? {
        didSet {
            localView?.device = metalDevice
            localView?.isPaused = true
            localView?.enableSetNeedsDisplay = false
            localView?.delegate = self
            localView?.framebufferOnly = false
            localView?.contentMode = .scaleAspectFit
        }
    }

    // Core image
    private var context: CIContext!
    private var lastImage: CIImage?

    // MARK: - Init

    init(videoSource: RTCVideoSource, videoCapturer: RTCVideoCapturer) {
        self.videoSource = videoSource
        self.videoCapturer = videoCapturer

        if #available(iOS 15, *) {
            self.supportsBackgroundBlur = true
        }

        super.init()

        initMetal()
        initVisionRequests()
        initAVCaptureSession()

        NotificationCenter.default.addObserver(self, selector: #selector(deviceOrientationDidChangeNotification), name: UIDevice.orientationDidChangeNotification, object: nil)
        self.updateVideoRotationBasedOnDeviceOrientation()
    }

    deinit {
        session?.stopRunning()
    }

    func initMetal() {
        metalDevice = MTLCreateSystemDefaultDevice()
        metalCommandQueue = metalDevice.makeCommandQueue()

        context = CIContext(mtlDevice: metalDevice)
    }

    func initVisionRequests() {
        if #available(iOS 15.0, *) {
            // Create a request to detect face rectangles.
            facePoseRequest = VNDetectFaceRectanglesRequest()
            facePoseRequest.revision = VNDetectFaceRectanglesRequestRevision3

            // Create a request to segment a person from an image.
            segmentationRequest = VNGeneratePersonSegmentationRequest()

            if let segmentationRequest = segmentationRequest as? VNGeneratePersonSegmentationRequest {
                segmentationRequest.qualityLevel = .balanced
                segmentationRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8
            }
        }
    }

    func initAVCaptureSession() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            fatalError("Error getting AVCaptureDevice.")
        }
        guard let input = try? AVCaptureDeviceInput(device: device) else {
            fatalError("Error getting AVCaptureDeviceInput")
        }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            self.session = AVCaptureSession()

            // TODO: Correctly determine AVCaptureDeviceFormat and fps (see ARDCaptureController.m)
            self.session?.sessionPreset = .vga640x480
            self.session?.addInput(input)

            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: .global(qos: .userInteractive))

            self.session?.addOutput(output)
            output.connections.first?.videoOrientation = .portrait
            self.session?.startRunning()
        }
    }

    // MARK: - Public switches

    public func enableBackgroundBlur() {
        DispatchQueue.global(qos: .userInteractive).async {
            self.backgroundBlurEnabled = true
        }
    }

    public func disableBackgroundBlur() {
        DispatchQueue.global(qos: .userInteractive).async {
            self.backgroundBlurEnabled = false
        }
    }

    // MARK: - Videoframe processing

    func blend(original frameImage: CIImage,
               mask maskPixelBuffer: CVPixelBuffer) -> CIImage? {

        // Create CIImage objects for the video frame and the segmentation mask.
        let originalImage = frameImage.oriented(.right)
        var maskImage = CIImage(cvPixelBuffer: maskPixelBuffer)

        // Scale the mask image to fit the bounds of the video frame.
        let scaleX = originalImage.extent.width / maskImage.extent.width
        let scaleY = originalImage.extent.height / maskImage.extent.height
        maskImage = maskImage.transformed(by: .init(scaleX: scaleX, y: scaleY))

        // Use "clampedToExtent()" to prevent black borders after applying the gaussian blur
        // Make sure to crop the image back afterwards to its original size, otherwise the result is disorted
        let backgroundImage = originalImage.clampedToExtent().applyingGaussianBlur(sigma: 10).cropped(to: originalImage.extent)

        // Blend the original, background, and mask images.
        let blendFilter = CIFilter.blendWithRedMask()
        blendFilter.inputImage = originalImage
        blendFilter.backgroundImage = backgroundImage
        blendFilter.maskImage = maskImage

        // Return the new blended image
        return blendFilter.outputImage?.oriented(.left)
    }

    func processVideoFrame(_ framePixelBuffer: CVPixelBuffer, _ sampleBuffer: CMSampleBuffer) {
        let pixelBuffer = framePixelBuffer
        var frameImage = CIImage(cvPixelBuffer: framePixelBuffer)

        if #available(iOS 15.0, *),
           self.backgroundBlurEnabled,
           let segmentationRequest = segmentationRequest as? VNGeneratePersonSegmentationRequest {

            // Perform the requests on the pixel buffer that contains the video frame.
            try? requestHandler.perform([facePoseRequest, segmentationRequest],
                                        on: pixelBuffer,
                                        orientation: .right)

            // Get the pixel buffer that contains the mask image.
            guard let maskPixelBuffer = segmentationRequest.results?.first?.pixelBuffer else {
                return
            }

            // Process the images.
            if let newImage = blend(original: frameImage, mask: maskPixelBuffer) {
                context.render(newImage, to: pixelBuffer)
                frameImage = newImage
            }
        }

        self.lastImage = frameImage
        self.localView?.draw()

        // Create the RTCVideoFrame
        let timeStampNs =  CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)) * Float64(NSEC_PER_SEC)
        let rtcpixelBuffer = RTCCVPixelBuffer(pixelBuffer: pixelBuffer)
        let videoFrame: RTCVideoFrame? = RTCVideoFrame(buffer: rtcpixelBuffer, rotation: videoRotation, timeStampNs: Int64(timeStampNs))

        if let videoFrame = videoFrame {
            self.videoSource.capturer(self.videoCapturer, didCapture: videoFrame)
        }
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = sampleBuffer.imageBuffer else {
            return
        }

        self.processVideoFrame(pixelBuffer, sampleBuffer)
    }

    // MARK: MTKViewDelegate

    func draw(in view: MTKView) {
        guard let commandBuffer = metalCommandQueue.makeCommandBuffer(),
              let currentDrawable = view.currentDrawable,
              let localView = localView,
              var ciImage = lastImage else {
            return
        }

        // Correctly rotate the local image
        if videoRotation == ._180 {
            ciImage = ciImage.oriented(.down)
        } else if videoRotation == ._90 {
            ciImage = ciImage.oriented(.right)
        } else if videoRotation == ._270 {
            ciImage = ciImage.oriented(.left)
        }

        // make sure the image is full screen
        let drawSize = localView.drawableSize
        let scaleX = drawSize.width / ciImage.extent.width
        let scaleY = drawSize.height / ciImage.extent.height

        var scale = scaleX

        // Make sure we use the smaller scale
        if scaleY < scaleX {
            scale = scaleY
        }

        // Make sure to scale by keeping the aspect ratio
        var newImage = ciImage.transformed(by: .init(scaleX: scale, y: scale))

        // render into the metal texture
        self.context.render(newImage,
                              to: currentDrawable.texture,
                              commandBuffer: commandBuffer,
                              bounds: newImage.extent,
                              colorSpace: CGColorSpaceCreateDeviceRGB())

        // register drawwable to command buffer
        commandBuffer.present(currentDrawable)
        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Delegate method not implemented.
    }

    // MARK: Notifications

    func deviceOrientationDidChangeNotification() {
        self.deviceOrientation = UIDevice.current.orientation
        self.updateVideoRotationBasedOnDeviceOrientation()
    }

    func updateVideoRotationBasedOnDeviceOrientation() {
        // Handle video rotation based on device orientation
        if deviceOrientation == .portrait {
            videoRotation = ._0
        } else if deviceOrientation == .portraitUpsideDown {
            videoRotation = ._180
        } else if deviceOrientation == .landscapeRight {
            videoRotation = usingFrontCamera ? ._270 : ._90
        } else if deviceOrientation == .landscapeLeft {
            videoRotation = usingFrontCamera ? ._90 : ._270
        }
    }
}
