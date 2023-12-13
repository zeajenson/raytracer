// #include <format>
// #include<vulkan/vulkan.h>
// #include<vector>
//
// static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
//     VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
//     VkDebugUtilsMessageTypeFlagsEXT messageType,
//     const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
//     void* pUserData) {
//
//     std::puts(std::format("validation layer :{} \n", pCallbackData->pMessage).c_str());
//
//     return VK_FALSE;
// }
//
// struct Vulkan_Stuff {
//     Vulkan_Stuff(){}
//     Vulkan_Stuff(Vulkan_Stuff const &) = delete;
//     Vulkan_Stuff(Vulkan_Stuff &&) = delete;
//     Vulkan_Stuff &operator=(Vulkan_Stuff const &) = delete;
//     Vulkan_Stuff &operator=(Vulkan_Stuff &&) = delete;
//
//     
//     // VkApplicationInfo app_info;
//     VkInstance instance;
//     VkDebugUtilsMessengerEXT debug_utils_messenger;
//
//
//     inline auto settup_debug_messanger() noexcept {}
//
//     inline auto settup() noexcept {
//
//         // auto debug_messenger_info = VkDebugUtilsMessengerCreateInfoEXT {
//         //     .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
//         //     .pNext = nullptr,
//         //     .flags = {},
//         //     .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
//         //     | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
//         //     | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
//         //     .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT 
//         //     | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT 
//         //     | VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT
//         //     | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
//         //     .pfnUserCallback = debug_callback,
//         // };
//         //
//         // auto vk_create_debug_utils_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(glfwGetInstanceProcAddress(instance, "vkCreateDebugUtilsMessengerEXT"));
//         // if(vk_create_debug_utils_messenger(instance, &debug_messenger_info, allocator, &debug_utils_messenger) not_eq VK_SUCCESS){
//         //     std::puts("unable to create debug utils messenger.\n");
//         // }
//     }
//
//     inline auto draw_frame() noexcept {}
// };
