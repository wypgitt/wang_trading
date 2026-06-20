import SwiftUI
import UIKit
import UserNotifications

// Push / notification layer: authorization, APNs registration, and the test path.
// Real APNs needs the Push Notifications capability + a server; everything here is
// verifiable on the simulator via local notifications or `xcrun simctl push`.
final class PushManager: ObservableObject {
    @Published var authStatus: UNAuthorizationStatus = .notDetermined

    func refreshStatus() {
        UNUserNotificationCenter.current().getNotificationSettings { settings in
            DispatchQueue.main.async { self.authStatus = settings.authorizationStatus }
        }
    }

    func requestAuthorization() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if let error { print("[Aperture] notification auth error: \(error.localizedDescription)") }
            DispatchQueue.main.async {
                self.refreshStatus()
                if granted { UIApplication.shared.registerForRemoteNotifications() }
            }
        }
    }

    /// Schedule a local notification mirroring an engine alert — lets you verify the
    /// notification → tap → deep-link flow without a live APNs server.
    func sendTestAlert(title: String, body: String, screen: String, symbol: String? = nil) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        var info: [String: Any] = ["screen": screen]
        if let symbol { info["symbol"] = symbol }
        content.userInfo = info
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: UNTimeIntervalNotificationTrigger(timeInterval: 2, repeats: false))
        UNUserNotificationCenter.current().add(request)
    }
}

// SwiftUI lifecycle needs a UIApplicationDelegate to receive APNs token callbacks
// and notification taps. Owns the shared Router / PushManager / session controller,
// which the App injects into the environment. @MainActor: all callbacks land on
// main, and it constructs the @MainActor session controller.
@MainActor
final class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {
    let router = Router()
    let push = PushManager()
    let session = SessionActivityController()

    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        UNUserNotificationCenter.current().delegate = self
        push.refreshStatus()
        // If a notification cold-launched the app, route it.
        if let info = launchOptions?[.remoteNotification] as? [AnyHashable: Any], let link = DeepLink.from(info) {
            router.handle(link)
        }
        return true
    }

    // APNs device token (production: POST to the BFF so the engine's alert path can
    // target this device). On an iOS 16+ simulator this still fires with a token.
    func application(_ application: UIApplication,
                     didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        let token = deviceToken.map { String(format: "%02x", $0) }.joined()
        print("[Aperture] APNs token: \(token)")
    }

    func application(_ application: UIApplication,
                     didFailToRegisterForRemoteNotificationsWithError error: Error) {
        print("[Aperture] APNs registration failed: \(error.localizedDescription)")
    }

    // Still surface the banner in the foreground.
    func userNotificationCenter(_ center: UNUserNotificationCenter,
                                willPresent notification: UNNotification) async -> UNNotificationPresentationOptions {
        [.banner, .sound, .list]
    }

    // Tap → deep-link into the relevant screen. (@MainActor: already on main.)
    func userNotificationCenter(_ center: UNUserNotificationCenter,
                                didReceive response: UNNotificationResponse) async {
        if let link = DeepLink.from(response.notification.request.content.userInfo) {
            router.handle(link)
        }
    }
}
