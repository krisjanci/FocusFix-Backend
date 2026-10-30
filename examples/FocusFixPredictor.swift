// Reference integration. Add this file and a validated .mlmodel to the app target.
// The app must compute the work-window features defined in README.md first.
import CoreML
import Foundation

final class FocusFixPredictor {
    static let featureOrder = [
        "elapsed_minutes", "work_fraction", "keys_per_minute", "deletes_per_minute",
        "clicks_per_minute", "switches_per_minute", "scrolls_per_minute",
        "mouse_points_per_minute", "mouse_events_per_minute", "mean_key_hold_seconds"
    ]
    enum PredictionError: Error { case incompatibleModel, invalidWindow, invalidFeatures, invalidOutput }
    private let model: MLModel
    let target: String
    let windowSeconds: Double

    init(url: URL, allowSynthetic: Bool = false) throws {
        let compiled = url.pathExtension == "mlmodel" ? try MLModel.compileModel(at: url) : url
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        model = try MLModel(contentsOf: compiled, configuration: configuration)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard metadata["feature_version"] == "focusfix_work_window_v1",
              let orderJSON = metadata["feature_order"]?.data(using: .utf8),
              let order = try? JSONDecoder().decode([String].self, from: orderJSON),
              order == Self.featureOrder,
              let window = Double(metadata["window_seconds"] ?? ""), window.isFinite, window >= 30,
              let output = metadata["target"], ["fatigue_0_10", "productivity_0_10"].contains(output),
              let synthetic = metadata["synthetic"]?.lowercased(), ["true", "false"].contains(synthetic),
              allowSynthetic || synthetic == "false" else {
            throw PredictionError.incompatibleModel
        }
        windowSeconds = window
        target = output
    }

    // Supply observed work seconds from complete mouse bins, including stationary
    // bins, before the prediction cutoff. Return nil when there is too little data.
    func predict(features: [String: Double], observedWorkSeconds: Double,
                 computedWindowSeconds: Double) throws -> Double? {
        guard computedWindowSeconds == windowSeconds, observedWorkSeconds.isFinite,
              observedWorkSeconds >= 0, observedWorkSeconds <= windowSeconds else {
            throw PredictionError.invalidWindow
        }
        if observedWorkSeconds < 30 { return nil }
        guard features.count == Self.featureOrder.count else { throw PredictionError.invalidFeatures }
        let input = try MLMultiArray(shape: [NSNumber(value: Self.featureOrder.count)], dataType: .double)
        for (index, name) in Self.featureOrder.enumerated() {
            guard let value = features[name], value.isFinite, value >= 0,
                  Float(value).isFinite else { throw PredictionError.invalidFeatures }
            input[index] = NSNumber(value: Double(Float(value)))
        }
        guard let fraction = features["work_fraction"], fraction <= 1 else {
            throw PredictionError.invalidFeatures
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["features": input])
        let output = try model.prediction(from: provider)
        guard let value = output.featureValue(for: "raw_rating"), value.type == .double,
              value.doubleValue.isFinite else { throw PredictionError.invalidOutput }
        return min(10, max(0, value.doubleValue))
    }
}
